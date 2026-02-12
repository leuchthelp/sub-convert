from dataclasses import dataclass
from colorama import Fore
import logging
import os 

#os.environ['TRANSFORMERS_OFFLINE'] = '1' 
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

logger = logging.getLogger(__name__)

static_languages = [
    "ar", "eu", "br", "ca", "zh", "Chinese_Hongkong", 
    "Chinese_Taiwan", "cv", "cs", "dv", "nl", "en", 
    "eo", "et", "fr", "fy", "ka", "de", "el", 
    "Hakha_Chin", "id", "ia", "it", "ja", "Kabyle", 
    "rw", "ky", "lv", "mt", "mn", "fa", "pl", 
    "pt", "ro", "Romansh_Sursilvan", "ru", "Sakha", "sl", 
    "es", "sv", "ta", "tt", "tr", "uk", "cy"
]


@dataclass
class ModelCore:

    def __init__(self):
        pass


@dataclass
class OCRModelCore(ModelCore):
    __slots__ = ("model", "processor", "torch_device")

    def __init__(
        self,
        model_name="PaddlePaddle/PaddleOCR-VL",
        options={},
    ):
        self.torch_device = options["torch_device"]

        attn_implementation="flash_attention_2"
        if "intel_disable_flash" in options or self.torch_device == "cpu":
            attn_implementation = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation, 
        ).to(device=self.torch_device).eval().share_memory() # type: ignore
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)


    def analyse(self, batch: list) ->  str:

        inputs = self.processor.apply_chat_template(
            batch, 
            add_generation_prompt=True,
	        tokenize=True,
	        return_dict=True,
	        return_tensors="pt",
            padding=True,
            padding_side='left',
        ).to(self.torch_device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)]
        texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        logger.debug(Fore.CYAN+ f"clean text: {texts}" + Fore.RESET)
        del inputs, generated_ids_trimmed, out

        return texts
    
    
    def __del__(self):
        del self.model
        del self.processor  


@dataclass
class LanguageModelCore(ModelCore):
    __slots__ = ("model", "processor", "torch_device", "languages", "tokenizer")


    def __init__(
        self,
        model_name="Mike0307/multilingual-e5-language-detection",
        languages=static_languages,
        options={}
    ):  
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=45)
        self.torch_device = options["torch_device"] if "torch_device" in options else "cpu"
        self.languages = languages

        self.model.to(self.torch_device)
        self.model.eval().share_memory()
        

    def __predict(self, text: str) -> torch.Tensor:
        tokenized = self.tokenizer(text.lower(), padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']


        with torch.no_grad():
            input_ids = input_ids.to(self.torch_device)
            attention_mask = attention_mask.to(self.torch_device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        logger.debug(Fore.MAGENTA + f"probabilities: {probabilities}" + Fore.RESET)

        del input_ids, attention_mask, logits, outputs, tokenized

        return probabilities
    

    def get_topk(self, text: str, k=3) -> list:
        
        probabilities = self.__predict(text=text)
        topk_prob, topk_indices = torch.topk(probabilities, k)

        topk_prob = topk_prob.cpu().numpy()[0].tolist()
        topk_indices = topk_indices.cpu().numpy()[0].tolist()

        topk_labels = [self.languages[index] for index in topk_indices]

        logger.debug(Fore.MAGENTA + f"top probilities: {topk_prob}, top labels: {topk_labels}" + Fore.RESET)

        tmp = [(a, b) for a, b in zip(topk_labels, topk_prob)]

        del probabilities, topk_labels, topk_prob, topk_indices

        return tmp
    

    def __del__(self):
        del self.model
        del self.tokenizer
    
