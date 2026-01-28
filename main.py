from dataclasses import dataclass
from pymkv import MKVTrack
from pysrt import SubRipFile, SubRipItem
from pathlib import Path
from collections import Counter
from colorama import Fore
from langcodes import *
from model_core import OCRModelCore, LanguageModelCore
from subtitle_track_manager import SubtitleTrackManager
from torch.multiprocessing import Pool, set_start_method
import numpy as np
import logging
import pytesseract as tess
import typing
import torch
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Runnable:

    def __init__(
            self,
            prompts: dict,
            task: str,
        ):

        self.fallback = False
        try:
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.fallback = True

        self.prompts = prompts
        self.task = task

        self.ocr_model      = OCRModelCore(torch_device=self.torch_device)
        self.language_model = LanguageModelCore(torch_device=self.torch_device)


    def run(self, sub_managers: list):
        for index, (image, item, track) in enumerate(sub_managers, start=1):
            items = []
            combined = []

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
                combined.append(self.language_model.get_topk(probabilities=probabilities))


            items.append(SubRipItem(index=index, start=item.start, end=item.end, text=text))

        track, SubRipFile(items=items), combined


def save_file(savable: typing.Tuple[MKVTrack, SubRipFile, list]):
    unique = 1
    for track, srt, combined in savable:
        if not combined:
            break

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

        logger.info(Fore.CYAN + f"{counter}, probablities {average}" + Fore.RESET)

        for label, count in counter.items():
            weights[label] = count / counter.total()

        for label, prob in average.items():
            average[label] = np.average(prob) * weights[label]

        logger.info(Fore.CYAN + f"{counter}, probablities {average}, weights: {weights}" + Fore.RESET)
        logger.info(average)

        final_lang = max(average, key=average.get)

        print(final_lang)

        path = path + (".sdh" if track.flag_hearing_impaired else "")
        path = path + (".forced" if track.forced_track else "")
        path = path + "." + (track.language if track.language_ietf == final_lang else Language.get(final_lang).to_alpha3() if track.language != None else "")

        potential_path = f"results/{path}.srt"

        logger.debug(f"path: {potential_path}, exists prior: {Path(potential_path).exists()}, global: {Path(potential_path).absolute()}")

        if Path(potential_path).exists():
            potential_path = potential_path.replace(path, f"{path}-{unique}")
            unique += 1

        srt.save(path=potential_path)


def main():
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    task = "ocr"
    prompts = {
        "ocr": "OCR:",
    }

    options = {
        "path_to_tmp": "tmp"
    }

    root = Path("test-files")
    convertibles = (path.absolute() for path in root.rglob("*") if not path.is_dir() and ".mkv" in path.name)
    sup_managers = sum([SubtitleTrackManager(file_path=path, options=options).get_pgs_managers() for path in convertibles],[])
    
    logger.info(sup_managers)

    try:
         set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    pool = Pool(processes=2)
    for result in pool.imap_unordered(Runnable(prompts=prompts, task=task).run, sup_managers):
        save_file(savable=result)
 
    
if __name__=="__main__":
    main()
