from torch.multiprocessing import Queue, Manager, Pool, set_start_method, current_process
from subtitle_track_manager import SubtitleTrackManager
from model_core import OCRModelCore, LanguageModelCore
from pysrt import SubRipFile, SubRipItem
from pgs_manager import PgsManager
from dataclasses import dataclass
from collections import Counter
from itertools import chain
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from pymkv import MKVTrack
from colorama import Fore
from pathlib import Path
from langcodes import *
import pytesseract as tess
import numpy as np
import argparse
import logging
import torch
import time
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Runnable:

    def __init__(
            self,
            prompts: dict,
            task: str,
            task_queue: Queue,
            progress_queue: Queue,
            options={},
        ):
        if not options:
            self.override_if_exists = False
        else:
            self.override_if_exists = options["override_if_exists"]

        self.fallback = False
        try:
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.torch_device == "cuda":

                # Check for working rocm and activate flash attention, otherwise its NVIDIA
                if torch.version.hip != None:
                    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

            if torch.xpu.is_available():
                options["intel_disable_flash"] = True
                self.torch_device = "xpu"
                
        except:
            self.fallback = True

        self.prompts = prompts
        self.task = task

        self.ocr_model      = OCRModelCore(torch_device=self.torch_device, options=options)
        self.language_model = LanguageModelCore(torch_device=self.torch_device)

        self.task_queue = task_queue
        self.progress_queue = progress_queue


    def run(self, pgs_manager: PgsManager) -> bool:
        
        pgs_data = pgs_manager.get_pgs_images()
        if not pgs_data:
            return False

        logger.debug(Fore.MAGENTA + f"Got to do {len(pgs_data)} items for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id} on {current_process()}" + Fore.RESET)
        
        self.task_queue.put_nowait((f"[cyan]{pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}", len(pgs_data)))

        savable = {"items": [], "combined": []}
        for index, (image, item, track) in enumerate(pgs_data, start=1):
            
            test_width, test_height = image.size

            if test_width == 0 or test_height == 0:
                return False

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

            self.progress_queue.put_nowait((f"[cyan]{pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}"))
        
        logger.debug(Fore.GREEN + f"Finished extracting and classifying for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}!" + Fore.RESET)
        
        self.save_file(savable=savable)
        return True


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
                potential_path = potential_path.replace(f"{path}", f"{path}-{unique}" if unique != 0 else f"{path}")
                potential_path = potential_path.replace(f"-{unique-1}", "",)
            
        srt.save(path=potential_path)


def main():
    parser = argparse.ArgumentParser(
        prog="PGS subtitle conversion using OCR and language identification on MKV files.",
        description="run python based PGS subtitle recognition",
    )
    parser.add_argument("-p", "--path", type=str, default="", help="Directory path to .mkv files. Will recursively scan subdirectories.")
    parser.add_argument("-o", "--override", type=bool, default=False, help="Override existing .srt file. Default: False")
    args = parser.parse_args()

    task = "ocr"
    prompts = {
        "ocr": "OCR:",
    }

    tmp_path = Path(f"{os.path.dirname(os.path.realpath(__file__))}/tmp")
    if tmp_path.exists() == False:
        tmp_path.mkdir()
    options = {
        "path_to_tmp": "tmp",
        "override_if_exists": args.override
    }

    root = Path("test-files")
    if args.path:
        root = Path(args.path)


    convertibles = (path.absolute() for path in root.rglob("*") if not path.is_dir() and ".mkv" in path.name)
    manager = Manager()
    task_queue = manager.Queue()
    progress_queue = manager.Queue()
    pgs_managers = chain.from_iterable((SubtitleTrackManager(file_path=path, options=options).get_pgs_managers() for path in convertibles))


    try:
         set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    runnable = Runnable(prompts=prompts, task=task, options=options, task_queue=task_queue, progress_queue=progress_queue)


    progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
    

    with progress:
        with Pool(processes=3) as pool:
            tasks = {}
            end = False
            for _ in pool.imap_unordered(runnable.run, pgs_managers):
                while end == False:
                    if task_queue.empty() == False:
                        description, total = task_queue.get_nowait()
                        task_id = progress.add_task(description=description, total=total, visible=True)

                        task = [task for task in progress.tasks if task.id == task_id][0]
                        tasks[description] = (task_id, task)


                    if progress_queue.empty() == False:
                        description = progress_queue.get_nowait()
                        if description in tasks:
                            task_id = tasks[description][0]
                            progress.update(task_id=task_id, advance=1)

                            task = tasks[description][1]

                            # Additionally have to check for if "task.remaining <= 1.0" as sometimes can get stuff with one missing
                            # Since file is still being saved and this only for fancy progressbar, should be ok
                            if task.finished or task.remaining <= 1.0:
                                progress.update(task_id=task.id, visible=True)


                    if progress.finished:
                        end = True
                      
    
if __name__=="__main__":
    main()
