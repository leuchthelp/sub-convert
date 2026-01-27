from dataclasses import dataclass
from typing import Generator
from colorama import Fore
from pymkv import  MKVTrack
from pathlib import Path
from media import Pgs
from PIL import Image
from media import PgsSubtitleItem
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
            options : dict,
    ):
        
        self.mkv_track = mkv_track
        self.options  = options
        self.hash     = hashlib.sha256(str(self.mkv_track).encode()).hexdigest()
        self.tmp_path = Path(f"{options["path_to_tmp"]}/{self.hash}")

        self.tmp_path.mkdir(parents=True)


    def get_pgs_images(self) -> Generator[Image, PgsSubtitleItem, MKVTrack]:
        tmp_file = f"{self.tmp_path}/{self.mkv_track.file_path}-{self.mkv_track.track_id}-{self.mkv_track.track_codec}.sup"
        cmd = ["mkvextract", self.mkv_track.file_path, "tracks", f"{self.mkv_track.track_id}:{tmp_file}"]

        subprocess.check_output(cmd)
        pgs = Pgs(data_reader=open(tmp_file, mode="rb").read)

        self.pgs_items = pgs.items

        return ((Image.fromarray(item.image.data), item, self.mkv_track) for item in self.pgs_items)
    

    def __del__(self):
        shutil.rmtree(path=self.tmp_path)

