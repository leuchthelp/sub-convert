from importlib.util import find_spec
from dataclasses import dataclass
from copy import deepcopy
import logging
import os

# os.environ['TRANSFORMERS_OFFLINE'] = '1'
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
            text = tess.image_to_string(
                image=entry, config="--oem 1 -l eng+deu+deu_frak+deu_latf+jpn"
            )

            # Small static fix for tesseract, might want to make this togglable in the future.
            # But realistically how often will | be used in subtitles?
            text = str(text).replace("|", "I")
            texts.append(text)

        return texts

    def __del__(self):
        del self


from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
import torch  # noqa: E402

from sub_convert2.utils.torch_utils import check_torch_cuda  # noqa: E402


@dataclass
class PaddleModelCore(OCRModelCore):
    __slots__ = ("model", "processor", "torch_device")

    def __init__(
        self,
        options: dict,
        model_name="PaddlePaddle/PaddleOCR-VL-1.6",
    ):
        super().__init__(options=options)
        self.torch_device = ""
        if options["torch_device"] is None or options["torch_device"] == "cuda":
            options = check_torch_cuda(options=options)

        self.torch_device = options["torch_device"]

        attn_implementation = "paged|sdpa"

        if find_spec("flash_attn") is not None and self.torch_device == "cuda":
            attn_implementation = "flash_attention_2"

        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(
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

        self.processor.tokenizer.padding_side = "left"
        
    def analyse(self, batch: list) -> list[str]:

        # Setup ocr prompt and message template
        ocr_task = "ocr"
        prompts = {
            "ocr": "OCR:",
        }
        message_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": prompts[ocr_task]},
                ],
            }
        ]

        messages = []
        for image in batch:
            tmp_template = deepcopy(message_template)
            tmp_template[0]["content"][0]["image"] = image
            messages.append(tmp_template)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(self.torch_device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False, use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)
        ]
        texts: list[str] = self.processor.post_process_image_text_to_text(
            generated_ids_trimmed
        )

        del inputs, generated_ids_trimmed, out
        return texts

    def __del__(self):
        del self.model
        del self.processor
