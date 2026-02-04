from torch.multiprocessing import current_process, Queue, Manager
from model_core import OCRModelCore, LanguageModelCore
from pysrt import SubRipFile, SubRipItem
from collections import Counter
from pgs_manager import PgsManager
from dataclasses import dataclass
from langcodes import Language
from pymkv import MKVTrack
from colorama import Fore
from pathlib import Path
import pytesseract as tess
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OCRGPUWorker:

    def __init__(
            self,
            messages: list,
            core: OCRModelCore,
            queues: dict[str, Queue],
            options={},
        ):
        self.process_queue  = queues["ocr_queue"]
        self.pass_queue     = queues["pass_queue"]
        self.options        = options
        self.torch_device   = self.options["torch_device"]

        self.core = core(options=self.options)
        self.core.model.to(self.torch_device).eval()
        self.messages = messages


    def run(self):
        end = False
        while end == False:
            if not self.process_queue.empty():
                image, return_queue = self.process_queue.get_nowait()

                if return_queue == -1:
                    self.pass_queue.put((None, -1))
                    end = True
                    continue
                else:
                    self.messages[0]["content"][0]["image"] = image
                    text = self.core.analyse(messages=self.messages)
                    self.pass_queue.put_nowait((text, return_queue))
            else:
                continue
    

@dataclass
class LangaugeGPUWorker:

    def __init__(
            self,
            core: LanguageModelCore,
            queues: dict[str, Queue],
            options={},
        ):
        self.pass_queue     = queues["pass_queue"]
        self.queues         = queues
        self.options        = options
        self.torch_device   = self.options["torch_device"]

        self.core = core(options=self.options)
        self.core.model.to(self.torch_device).eval()


    def run(self):
        end = False
        while end == False:
            if not self.pass_queue.empty():
                original_text, return_queue = self.pass_queue.get_nowait()

                if return_queue == -1:
                    end = True
                    continue
                else:
                    text = str(original_text).lower()
                    combined = self.core.get_topk(text=text)
                    self.queues[return_queue].put_nowait((original_text, combined))
            else:
                continue
        
        
@dataclass
class CPUWorker:

    def __init__(
            self,
            queues: dict[str, Queue],
            options={},
        ):
        self.gpu_ocr_queue      = queues["ocr_queue"]
        self.queues             = queues
        # Literally just for the progressbars to function as expected
        self.task_queue         = queues["task_queue"]
        self.progress_queue     = queues["progress_queue"]
        self.fallback           = options["fallback_status"]
        self.override_if_exists = options["override_if_exists"]


    def run(self, pgs_manager: PgsManager) -> bool:
        
        pgs_data = pgs_manager.get_pgs_images()
        if not pgs_data:
            return False

        logger.debug(Fore.MAGENTA + f"Got to do {len(pgs_data)} items for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id} on {current_process()}" + Fore.RESET)
        
        self.task_queue.put_nowait((f"[cyan]{pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}", len(pgs_data)))

        queue_index = current_process().name.split("-")[1]
        return_queue = self.queues[f"{queue_index}"]
        finished = []
        for image, item, track in pgs_data:
            
            test_width, test_height = image.size
            if test_width == 0 or test_height == 0:
                continue

            
            image = image.convert("RGB")
            text = ""
            if self.fallback == False:
                self.gpu_ocr_queue.put_nowait((image, queue_index))
            else: 
                text = tess.image_to_string(image=image)
            
            finished.append((item, track, text))

            

        savable = {"items": [], "combined": []}  
        for index, (item, track, text) in enumerate(finished, start=1):
            self.progress_queue.put_nowait((f"[cyan]{pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}"))
            
            combined = []
            if self.fallback == False:
                text, combined = return_queue.get()

            if not text:
                continue

            sub_item = SubRipItem(index=index, start=item.start, end=item.end, text=text)
            
            savable["track"] = track
            savable["items"].append(sub_item)

            if self.fallback == False:
                savable["combined"].append(combined)
        
        logger.debug(Fore.GREEN + f"Finished extracting and classifying for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}!" + Fore.RESET)
        
        self.save_file(savable=savable)
        return True


    def save_file(self, savable: dict[str, MKVTrack | list]):
        items    = savable["items"]
        srt      = SubRipFile(items=items)

        if "track" not in savable:
            return
        
        combined = []
        if self.fallback == False:
            combined = savable["combined"]
        
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

        final_lang = track.language_ietf
        if self.fallback == False:
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

