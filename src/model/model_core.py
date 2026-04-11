from dataclasses import dataclass
from colorama import Fore
import logging
import typing
import torch
import os

# os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

static_languages = [
    "ar",
    "eu",
    "br",
    "ca",
    "zh",
    "Chinese_Hongkong",
    "Chinese_Taiwan",
    "cv",
    "cs",
    "dv",
    "nl",
    "en",
    "eo",
    "et",
    "fr",
    "fy",
    "ka",
    "de",
    "el",
    "Hakha_Chin",
    "id",
    "ia",
    "it",
    "ja",
    "Kabyle",
    "rw",
    "ky",
    "lv",
    "mt",
    "mn",
    "fa",
    "pl",
    "pt",
    "ro",
    "Romansh_Sursilvan",
    "ru",
    "Sakha",
    "sl",
    "es",
    "sv",
    "ta",
    "tt",
    "tr",
    "uk",
    "cy",
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
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        options={},
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch_device = options["torch_device"]

        attn_implementation = "flash_attention_2"
        if "intel_disable_flash" in options or self.torch_device == "cpu":
            attn_implementation = "sdpa"

        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map="auto",
            )
            .to(device=self.torch_device)  # type: ignore
            .eval()
            .share_memory()
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, backend="torchvision"
        )

    def analyse(self, batch: list) -> list[str]:

        inputs = self.processor.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.torch_device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False, use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)
        ]
        texts: list[str] = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        logger.debug(Fore.CYAN + f"clean text: {texts}" + Fore.RESET)
        del inputs, generated_ids_trimmed, out

        return texts

    def __del__(self):
        del self.model
        del self.processor


@dataclass
class TesseractCore(OCRModelCore):
    def __init__(self, options={}):
        pass

    def analyse(self, batch: list) -> list[str]:
        import pytesseract as tess

        texts: list[str] = []
        for entry in batch:
            image = entry[0]["content"][0]["image"]
            texts.append(tess.image_to_string(image=image))

        return texts
    
    def __del__(self):
        del self


@dataclass
class LanguageModelCore(ModelCore):
    __slots__ = ("model", "tokenizer", "torch_device", "languages")

    def __init__(
        self,
        model_name="Mike0307/multilingual-e5-language-detection",
        languages: list[str] = static_languages,
        options={},
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name
        )

        self.torch_device = (
            options["torch_device"] if "torch_device" in options else "cpu"
        )

        attn_implementation = "flash_attention_2"
        if "intel_disable_flash" in options or self.torch_device == "cpu":
            attn_implementation = "sdpa"

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=45,
                dtype=torch.float16,
                attn_implementation=attn_implementation,
                device_map="auto",
            )
            .to(self.torch_device)
            .eval()
            .share_memory()
        )

        self.languages = languages

    def __predict(self, text: str) -> torch.Tensor:
        tokenized = self.tokenizer(
            text.lower(),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.torch_device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        logger.debug(Fore.MAGENTA + f"probabilities: {probabilities}" + Fore.RESET)

        del logits, outputs, tokenized

        return probabilities

    def get_topk(self, text: str, k=3) -> list[tuple[str, typing.Any]]:

        probabilities = self.__predict(text=text)
        topk_prob, topk_indices = torch.topk(probabilities, k)

        topk_prob = topk_prob.cpu().numpy()[0].tolist()
        topk_indices = topk_indices.cpu().numpy()[0].tolist()

        topk_labels: list[str] = [self.languages[index] for index in topk_indices]

        logger.debug(
            Fore.MAGENTA
            + f"top probilities: {topk_prob}, top labels: {topk_labels}"
            + Fore.RESET
        )

        tmp = [(a, b) for a, b in zip(topk_labels, topk_prob)]

        del probabilities, topk_labels, topk_prob, topk_indices

        return tmp

    def __del__(self):
        del self.model
        del self.tokenizer


@dataclass
class LinguaCore(LanguageModelCore):
    __slots__ = "detector"

    def __init__(
        self,
        options={},
    ):
        self.detector = None

    def __init_around_pickle(self):
        from lingua import Language, LanguageDetectorBuilder

        languages = [
            Language.ENGLISH,
            Language.FRENCH,
            Language.GERMAN,
            Language.SPANISH,
            Language.JAPANESE,
        ]
        return LanguageDetectorBuilder.from_languages(*languages).build()

    def get_topk(self, text: str, k=3) -> list[tuple[str, typing.Any]]:
        if self.detector is None:
            self.detector = self.__init_around_pickle()

        confidence_values = self.detector.compute_language_confidence_values(text)
        tmp = [
            (str(confidence.language.iso_code_639_1.name), float(confidence.value))
            for confidence in confidence_values
        ]
        return tmp

    def __del__(self):
        del self
