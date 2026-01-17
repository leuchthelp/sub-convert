from pymkv import MKVFile, MKVTrack
from media import Pgs
from subprocess import check_output
from PIL import Image
from pysrt import SubRipFile, SubRipItem
from pathlib import Path
import logging
import pytesseract as tess
import typing


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
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
