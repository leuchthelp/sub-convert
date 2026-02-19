from transformers import AutoModelForCausalLM, AutoProcessor
from copy import deepcopy
from PIL import Image
import numpy as np
import torch


def main():
    # Model setup
    torch_device        = "cuda"
    model_name          = "PaddlePaddle/PaddleOCR-VL"
    attn_implementation = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation, 
        ).to(device=torch_device).eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)


    # Batch setup
    batch = []
    batch_size = 64
    task = "ocr"
    prompts = {
        "ocr": "OCR:",
    }
    message_template = [
                {"role": "user",         
                 "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": prompts[task]},
                    ]
                }
            ]

    for _ in range(0, batch_size):
        height  = np.random.randint(24, 180)
        width   = np.random.randint(24, 180)
        a       = np.random.rand(height, width, 3) * 255
        image   = Image.fromarray(a.astype('uint8')).convert('RGB')

        message = deepcopy(message_template)
        message[0]["content"][0]["image"] = image
        batch.append(message)


    # Input setup
    inputs = processor.apply_chat_template(
            batch, 
            add_generation_prompt=True,
	        tokenize=True,
	        return_dict=True,
	        return_tensors="pt",
            padding=True,
            padding_side='left',
        ).to(torch_device)
    
    with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)]
    texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)



if __name__ == "__main__":
    main()