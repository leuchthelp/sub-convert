from dataclasses import dataclass
import logging
import os

from colorama import Fore


# os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class OCRModelCore:
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


import torch  # noqa: E402

from src.utils.torch_utils import check_torch_cuda  # noqa: E402


@dataclass
class PaddleModelCore(OCRModelCore):
    __slots__ = ("model", "processor", "torch_device")

    def __init__(
        self,
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        options={},
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        options = check_torch_cuda(options=options)
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
