from dataclasses import dataclass
from pymkv import MKVTrack
from pysrt import SubRipFile, SubRipItem
from pathlib import Path
from collections import Counter
from colorama import Fore
from langcodes import *
from model_core import OCRModelCore, LanguageModelCore
from subtitle_track_manager import SubtitleTrackManager
from torch.multiprocessing import Pool, set_start_method, current_process
from itertools import chain
from pgs_manager import PgsManager
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
import numpy as np
import logging
import pytesseract as tess
import torch
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubtitleProgressBar(Progress):

    def __init__(self):
        pass


@dataclass
class Runnable:

    def __init__(
            self,
            prompts: dict,
            task: str,
            options={},
        ):
        if not options:
            self.override_if_exists = False
        else:
            self.override_if_exists = options["override_if_exists"]

        self.fallback = False
        try:
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.fallback = True

        self.prompts = prompts
        self.task = task

        self.ocr_model      = OCRModelCore(torch_device=self.torch_device)
        self.language_model = LanguageModelCore(torch_device=self.torch_device)

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )


    def run(self, pgs_manager: PgsManager) -> dict[str, MKVTrack | list]:
        
        pgs_data = pgs_manager.get_pgs_images()
        if not pgs_data:
            return {}

        logger.info(Fore.MAGENTA + f"Got to do {len(pgs_data)} items for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id} on {current_process()}" + Fore.RESET)

        savable = {"items": [], "combined": []}
        for index, (image, item, track) in enumerate(pgs_data, start=1):
            messages = [
                {"role": "user",         
                 "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": self.prompts[self.task]},
                    ]
                }
            ]

            text = self.ocr_model.analyse(messages=messages)

            if self.fallback:
                text = tess.image_to_string(image=image)
            else:
                probabilities = self.language_model.predict(text=text)
                combined = self.language_model.get_topk(probabilities=probabilities)


            sub_item = SubRipItem(index=index, start=item.start, end=item.end, text=text)
            
            savable["track"] = track
            savable["items"].append(sub_item)
            savable["combined"].append(combined)
        
        logger.info(Fore.GREEN + f"Finished extracting and classifying for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}!" + Fore.RESET)
        return savable


    def save_file(self, savable: dict[str, MKVTrack | list]):
        combined = savable["combined"]
        items    = savable["items"]
        srt      = SubRipFile(items=items)
        track    = savable["track"]

        path = Path(track.file_path).name.replace(".mkv", "")
        counter = Counter()
        average = {}
        weights = {}

        for both in combined:
            for label, prob in both:
                counter.update([label])
                if label not in average:
                    average[label] = [prob]
                else:
                    average[label].append(prob)

        logger.debug(Fore.CYAN + f"{counter}, probablities {average}" + Fore.RESET)

        for label, count in counter.items():
            weights[label] = count / counter.total()
        for label, prob in average.items():
            average[label] = np.average(prob) * weights[label]

        logger.debug(Fore.CYAN + f"{counter}, probablities {average}, weights: {weights}" + Fore.RESET)

        final_lang = max(average, key=average.get)

        logger.debug(Fore.MAGENTA + f"picked language: {final_lang}, averages: {average}" + Fore.RESET)

        forced = True if track.forced_track or len(items) <= 150 else False

        path = path + (".sdh" if track.flag_hearing_impaired else "")
        path = path + (".forced" if forced else "")
        path = path + "." + (track.language if track.language_ietf == final_lang else Language.get(final_lang).to_alpha3() if track.language != None else "")
        potential_path = f"{Path(track.file_path).parent}/{path}.srt"

        logger.debug(f"path: {potential_path}, exists prior: {Path(potential_path).exists()}, global: {Path(potential_path).absolute()}")

        logger.debug(f"{self.override_if_exists == True} and {Path(potential_path)} exists: {Path(potential_path).exists()}")
        if self.override_if_exists == True and Path(potential_path).exists():
            Path(potential_path).unlink()
        else:
            unique = 0
            while Path(potential_path).exists():
                unique += 1
            potential_path = potential_path.replace(path, f"{path}-{unique}" if unique != 0 else f"{path}")
            
        srt.save(path=potential_path)


def main():
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

    task = "ocr"
    prompts = {
        "ocr": "OCR:",
    }

    options = {
        "path_to_tmp": "tmp",
        "override_if_exists": True
    }

    root = Path("test-files")
    convertibles = (path.absolute() for path in root.rglob("*") if not path.is_dir() and ".mkv" in path.name)
    pgs_data = chain.from_iterable((SubtitleTrackManager(file_path=path, options=options).get_pgs_managers() for path in convertibles))

    try:
         set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    runnable = Runnable(prompts=prompts, task=task, options=options)

    pool = Pool(processes=6)
    for result in pool.imap_unordered(runnable.run, pgs_data):
        if not result:
            continue
        runnable.save_file(savable=result)
 
    
if __name__=="__main__":
    main()
