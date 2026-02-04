from dataclasses import dataclass
from colorama import Fore
from pymkv import  MKVTrack, MKVFile
from media import PgsSubtitleItem
from pathlib import Path
from media import Pgs
from PIL import Image
import subprocess
import logging
import hashlib
import shutil
logger = logging.getLogger(__name__)


@dataclass
class PgsManager:

    def __init__(
            self,
            mkv_track: MKVTrack,
            mkv_file: MKVFile,
            options : dict,
    ):
        
        self.mkv_track= mkv_track
        self.mkv_file = mkv_file
        self.options  = options
        self.hash     = hashlib.sha256(str(self.mkv_track).encode()).hexdigest()
        self.tmp_path = Path(f"{options["path_to_tmp"]}/{self.hash}")

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self.tmp_path.mkdir(parents=True)


    def get_pgs_images(self) -> list[Image, PgsSubtitleItem, MKVTrack]:
        tmp_file = f"{self.tmp_path}/{self.mkv_track.file_path}-{self.mkv_track.track_id}-{self.mkv_track.track_codec}.sup"
        cmd = ["mkvextract", self.mkv_track.file_path, "tracks", f"{self.mkv_track.track_id}:{tmp_file}"]

        subprocess.check_output(cmd)
        pgs = Pgs(data_reader=open(tmp_file, mode="rb").read)

        self.pgs_items = pgs.items

        shutil.rmtree(path=self.tmp_path)

        return [(Image.fromarray(item.image.data), item, self.mkv_track) for item in self.pgs_items]

