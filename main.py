from pymkv import MKVFile, MKVTrack
from media import Pgs
from subprocess import check_output
from PIL import Image
from pysrt import SubRipFile, SubRipItem
from pathlib import Path
from collections import Counter
from colorama import Fore
from langcodes import *
import numpy as np
import logging
import pytesseract as tess
import typing

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#languages = [
#    "Arabic", "Basque", "Breton", "Catalan", "Chinese_China", "Chinese_Hongkong", 
#    "Chinese_Taiwan", "Chuvash", "Czech", "Dhivehi", "Dutch", "English", 
#    "Esperanto", "Estonian", "French", "Frisian", "Georgian", "German", "Greek", 
#    "Hakha_Chin", "Indonesian", "Interlingua", "Italian", "Japanese", "Kabyle", 
#    "Kinyarwanda", "Kyrgyz", "Latvian", "Maltese", "Mongolian", "Persian", "Polish", 
#    "Portuguese", "Romanian", "Romansh_Sursilvan", "Russian", "Sakha", "Slovenian", 
#    "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Ukranian", "Welsh"
#]

languages = [
    "ar", "eu", "br", "ca", "zh", "Chinese_Hongkong", 
    "Chinese_Taiwan", "cv", "cs", "dv", "nl", "en", 
    "eo", "et", "fr", "fy", "ka", "de", "el", 
    "Hakha_Chin", "id", "ia", "it", "ja", "Kabyle", 
    "rw", "ky", "lv", "mt", "mn", "fa", "pl", 
    "pt", "ro", "Romansh_Sursilvan", "ru", "Sakha", "sl", 
    "es", "sv", "ta", "tt", "tr", "uk", "cy"
]


def predict(text, model, tokenizer, device = torch.device('cpu')):
    model.to(device)
    model.eval()
    tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities


def get_topk(probabilities, languages, k=3):
    topk_prob, topk_indices = torch.topk(probabilities, k)
    topk_prob = topk_prob.cpu().numpy()[0].tolist()
    topk_indices = topk_indices.cpu().numpy()[0].tolist()
    topk_labels = [languages[index] for index in topk_indices]
    return topk_prob, topk_labels


def main():
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

    model_path = "PaddlePaddle/PaddleOCR-VL"
    task = "ocr"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PROMPTS = {
        "ocr": "OCR:",
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
    ).to(dtype=torch.bfloat16, device=DEVICE).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    fallback = False

    tokenizer = AutoTokenizer.from_pretrained('Mike0307/multilingual-e5-language-detection')
    model_lang = AutoModelForSequenceClassification.from_pretrained('Mike0307/multilingual-e5-language-detection', num_labels=45)


    mkv = MKVFile("test-files/Arcane S01E01.mkv")

    savable: typing.List[typing.Tuple[MKVTrack, SubRipFile, list]] = []

    tracks = mkv.tracks

    for track in tracks:
        if track.track_type == "subtitles": 
            items = []
            combined = []
            
            tmp_file = f"tmp/{track.track_id}-{track.track_codec}.sup"
            cmd = ['mkvextract', track.file_path, 'tracks', f'{track.track_id}:{tmp_file}']
            check_output(cmd)
            pgs = Pgs(data_reader=open(tmp_file, mode="rb").read, temp_folder="tmp")

            test = pgs.items
            
            for index, item in enumerate(test, start=1):
                
                image = Image.fromarray(item.image.data)
                image.save(f"tmp/{track.file_id}-{index}.png")

                messages = [
                    {"role": "user",         
                     "content": [
                            {"type": "image", "image": Image.open(f"tmp/{track.file_id}-{index}.png").convert("RGB")},
                            {"type": "text", "text": PROMPTS[task]},
                        ]
                    }
                ]
                
                inputs = processor.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 	
                    return_dict=True,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        use_cache=True
                    )

                text = processor.batch_decode(out, skip_special_tokens=True)[0]
                text = str(text).partition("Assistant: ")[2].replace("\"", "")
                
                probabilities = predict(text, model_lang, tokenizer, torch.device("cuda"))
                topk_prob, topk_labels = get_topk(probabilities, languages)

                combined.append([(a, b) for a, b in zip(topk_labels, topk_prob)])


                logger.info(Fore.GREEN + f"text: {text} and detected language: {(topk_labels, topk_prob)}" + Fore.RESET)

                if fallback:
                    text = tess.image_to_string(image=Image.fromarray(item.image.data))
                
                items.append(SubRipItem(index=index, start=item.start, end=item.end, text=text))
                

            savable.append((track, SubRipFile(items=items), combined))
    

    unique = 1
    for track, srt, combined in savable:
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

        logger.info(average)
        
        max_key = None
        max_value = float('-inf')

        for language, probability in average.items():
            if probability > max_value:
                max_value = probability
                max_key = language

        final_lang = max_key

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
    
    
    
    
if __name__=="__main__":
    main()
