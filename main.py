from pymkv import MKVFile, MKVTrack
from media import Pgs
from subprocess import check_output
from PIL import Image
from pysrt import SubRipFile, SubRipItem
from pathlib import Path
import logging
import pytesseract as tess
import typing

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import os


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

    llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
    trust_remote_code=True
    )
    prompt = "<|vision_start|><|image_pad|><|vision_end|>\nFree OCR."

    mkv = MKVFile("test-files/The Seven Deadly Sins S01E10.mkv")

    savable: typing.List[typing.Tuple[MKVTrack, SubRipFile]] = []
    
    for track in mkv.tracks:
        if track.track_type == "subtitles": 
            items = []
            
            tmp_file = f"tmp/{track.track_id}-{track.track_codec}.sup"
            cmd = ['mkvextract', track.file_path, 'tracks', f'{track.track_id}:{tmp_file}']
            check_output(cmd)
            pgs = Pgs(data_reader=open(tmp_file, mode="rb").read, temp_folder="tmp")

            test = pgs.items[0:10]
            
            for index, item in enumerate(test, start=1):
                
                image = Image.fromarray(item.image.data)
                image.save(f"tmp/{track.file_id}-{index}.png")
                image = Image.open(f"tmp/{track.file_id}-{index}.png")

                model_input = [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image}
                    }
                ]

                sampling_param = SamplingParams(
                            temperature=0.0,
                            max_tokens=8192,
                            # ngram logit processor args
                            extra_args=dict(
                                ngram_size=30,
                                window_size=90,
                                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
                            ),
                            skip_special_tokens=False,
                )
                model_outputs = llm.generate(model_input, sampling_param)

                for output in model_outputs:
                    logger.info(f"text: {output.outputs[0].text}")

                
                text = tess.image_to_string(image=Image.fromarray(item.image.data))
                
                items.append(SubRipItem(index=index, start=item.start, end=item.end, text=text))
                

            savable.append((track, SubRipFile(items=items)))
    

    unique = 1
    for track, srt in savable:
        path = Path(track.file_path).name.replace(".mkv", "")

        
        path = path + (".sdh" if track.flag_hearing_impaired else "")
        path = path + (".forced" if track.forced_track else "")
        path = path + ("." + track.language if track.language != None else "")
        
        potential_path = f"results/{path}.srt"
        
        
        logger.info(f"path: {potential_path}, exists prior: {Path(potential_path).exists()}, global: {Path(potential_path).absolute()}")
        
        if Path(potential_path).exists():
            potential_path = potential_path.replace(path, f"{path}-{unique}")
            unique += 1
        
        srt.save(path=potential_path)
    
    
    
    
if __name__=="__main__":
    main()
