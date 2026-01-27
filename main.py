from pymkv import MKVTrack
from pysrt import SubRipFile, SubRipItem
from pathlib import Path
from collections import Counter
from colorama import Fore
from langcodes import *
from model_core import OCRModelCore, LanguageModelCore
from subtitle_track_manager import SubtitleTrackManager
from itertools import chain
import numpy as np
import logging
import pytesseract as tess
import typing
import torch
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_file(savable: typing.List[typing.Tuple[MKVTrack, SubRipFile, list]]):
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

    fallback = False
    try:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        fallback = True

    task = "ocr"

    PROMPTS = {
        "ocr": "OCR:",
    }
    ocr_model      = OCRModelCore(torch_device=DEVICE)
    language_model = LanguageModelCore(torch_device=DEVICE)


    options = {
        "path_to_tmp": "tmp"
    }

    savable: typing.List[typing.Tuple[MKVTrack, SubRipFile, list]] = []

    root = Path("test-files")
    convertibles = (path.absolute() for path in root.rglob("*") if not path.is_dir() and ".mkv" in path.name)
    sub_managers = (chain.from_iterable(SubtitleTrackManager(file_path=path, options=options).get_pgs_managers_data()) for path in convertibles) 

    
    items = []
    combined = []
    
    
    for index, (image, item, track) in enumerate(chain.from_iterable(sub_managers), start=1):
        
        messages = [
            {"role": "user",         
             "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": PROMPTS[task]},
                ]
            }
        ]
        
        text = ocr_model.analyse(messages=messages)
        
        probabilities = language_model.predict(text=text)
        combined.append(language_model.get_topk(probabilities=probabilities))
        if fallback:
            text = tess.image_to_string(image=image)
        
        items.append(SubRipItem(index=index, start=item.start, end=item.end, text=text))
        
    savable.append((track, SubRipFile(items=items), combined))

    save_file(savable=savable)
 
    
if __name__=="__main__":
    main()
