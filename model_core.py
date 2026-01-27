from dataclasses import dataclass
from colorama import Fore
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

languages = [
    "ar", "eu", "br", "ca", "zh", "Chinese_Hongkong", 
    "Chinese_Taiwan", "cv", "cs", "dv", "nl", "en", 
    "eo", "et", "fr", "fy", "ka", "de", "el", 
    "Hakha_Chin", "id", "ia", "it", "ja", "Kabyle", 
    "rw", "ky", "lv", "mt", "mn", "fa", "pl", 
    "pt", "ro", "Romansh_Sursilvan", "ru", "Sakha", "sl", 
    "es", "sv", "ta", "tt", "tr", "uk", "cy"
]

@dataclass
class LanguageModelCore:

    def __init__(
        self,
        model_name="Mike0307/multilingual-e5-language-detection",
        torch_device="cpu",
        languages=languages

    ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=45)
        self.torch_device = torch.device(torch_device)
        self.languages = languages
        

    def predict(self, text: str) -> torch.Tensor:
        self.model.to(self.torch_device)
        self.model.eval()
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']


        with torch.no_grad():
            input_ids = input_ids.to(self.torch_device)
            attention_mask = attention_mask.to(self.torch_device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        logger.debug(Fore.MAGENTA + f"probabilities: {probabilities}" + Fore.RESET)

        return probabilities
    

    def get_topk(self, probabilities, k=3) -> list:
        topk_prob, topk_indices = torch.topk(probabilities, k)

        topk_prob = topk_prob.cpu().numpy()[0].tolist()
        topk_indices = topk_indices.cpu().numpy()[0].tolist()

        topk_labels = [self.languages[index] for index in topk_indices]

        logger.debug(Fore.MAGENTA + f"top probilities: {topk_prob}, top labels: {topk_labels}" + Fore.RESET)

        return [(a, b) for a, b in zip(topk_labels, topk_prob)]
    

@dataclass
class OCRModelCore:

    def __init__(
        self,
        model_name="PaddlePaddle/PaddleOCR-VL",
        torch_device="cpu",

    ):
        self.torch_device = torch_device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", 
        ).to(dtype=torch.bfloat16, device=self.torch_device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    def analyse(self, messages: list) ->  str:

        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 	
            return_dict=True,
            return_tensors="pt"
        ).to(self.torch_device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )

        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        text = str(text).partition("Assistant: ")[2]

        logger.info(Fore.CYAN+ f"clean text: {text}" + Fore.RESET)

        return text
