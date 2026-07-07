from importlib.util import find_spec
from dataclasses import dataclass
import logging
import typing
import os

# os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STATIC_LANGUAGES = [
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
class LanguageModelCore:
    __slots__ = ("detector", "options")

    from lingua import Language, LanguageDetectorBuilder, LanguageDetector

    def __init__(
        self,
        options: dict,
    ):
        self.options = options
        self.detector = None

    def __init_around_pickle(self) -> LanguageDetector:
        languages = [
            self.Language.ENGLISH,
            self.Language.FRENCH,
            self.Language.GERMAN,
            self.Language.SPANISH,
            self.Language.JAPANESE,
        ]
        return self.LanguageDetectorBuilder.from_languages(*languages).build()

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


from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: E402
import torch  # noqa: E402

from sub_convert2.utils.torch_utils import check_torch_cuda  # noqa: E402


@dataclass
class LangDetectModelCore(LanguageModelCore):
    __slots__ = ("model", "tokenizer", "torch_device", "languages")

    def __init__(
        self,
        options: dict,
        model_name="Mike0307/multilingual-e5-language-detection",
        languages=None,
    ):
        super().__init__(options=options)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name
        )

        self.torch_device = ""
        if options["torch_device"] is None or options["torch_device"] == "cuda":
            options = check_torch_cuda(options=options)

        self.torch_device = options["torch_device"]

        attn_implementation = "paged|sdpa"
        if find_spec("flash_attn") is not None and self.torch_device == "cuda":
            attn_implementation = "flash_attention_2"

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
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

        if languages is None:
            languages = STATIC_LANGUAGES
        self.languages = languages

    def __predict(self, text: str) -> torch.Tensor:
        tokenized = self.tokenizer(
            text.lower(),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.torch_device)  # ty:ignore[call-non-callable]

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        del logits, outputs, tokenized

        return probabilities

    def get_topk(self, text: str, k=3) -> list[tuple[str, typing.Any]]:

        probabilities = self.__predict(text=text)
        topk_prob, topk_indices = torch.topk(probabilities, k)

        topk_prob = topk_prob.cpu().numpy()[0].tolist()
        topk_indices = topk_indices.cpu().numpy()[0].tolist()

        topk_labels: list[str] = [self.languages[index] for index in topk_indices]
        tmp = list(zip(topk_labels, topk_prob))

        del probabilities, topk_labels, topk_prob, topk_indices

        return tmp

    def __del__(self):
        del self.model
        del self.tokenizer
