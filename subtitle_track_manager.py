from dataclasses import dataclass
from colorama import Fore
from pymkv import MKVFile, MKVTrack
from media import PgsSubtitleItem
from PIL import Image
from typing import Generator
from pgs_manager import PgsManager
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubtitleTrackManager:

    def __init__(
            self,
            file_path: str,
            options: dict,
    ):
        self.mkv_file = MKVFile(file_path=file_path)
        self.tracks = (track for track in self.mkv_file.tracks if track.track_type == "subtitles")
        self.options = options


    def get_pgs_managers_data(self) -> Generator[Image, PgsSubtitleItem, MKVTrack]:
        return (PgsManager(mkv_track=track, options=self.options).get_pgs_images() for track in self.tracks)


