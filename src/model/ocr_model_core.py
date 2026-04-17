from importlib.util import find_spec
from dataclasses import dataclass
import logging
import os

#os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class OCRModelCore:
    def __init__(self, options: dict):
        self.options = options

    def analyse(self, batch: list) -> list[str]:
        import pytesseract as tess

        texts: list[str] = []
        for entry in batch:
            image = entry[0]["content"][0]["image"]
            text = tess.image_to_string(
                image=image, config="--oem 1 -l eng+deu+deu_frak+deu_latf+jpn"
            )

            # Small static fix for tesseract, might want to make this togglable in the future.
            # But realistically how often will | be used in subtitles?
            text = str(text).replace("|", "I")
            texts.append(text)

        return texts

    def __del__(self):
        del self


import torch  # noqa: E402

from src.utils.torch_utils import check_torch_cuda  # noqa: E402


@dataclass
class PaddleModelCore(OCRModelCore):
    __slots__ = ("model", "processor", "torch_device")

    def __init__(
        self,
        options: dict,
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    ):
        super().__init__(options=options)

        from transformers import AutoModelForImageTextToText, AutoProcessor

        options = check_torch_cuda(options=options)
        self.torch_device = options["torch_device"]

        attn_implementation = "paged|sdpa"
        if find_spec("flash_attn") is not None:
            attn_implementation = "flash_attention_2"


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

        del inputs, generated_ids_trimmed, out
        return texts

    def __del__(self):
        del self.model
        del self.processor
