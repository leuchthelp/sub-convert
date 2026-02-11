from pgs_manager import PgsManager
from dataclasses import dataclass
from pymkv import MKVFile
from pathlib import Path
import logging
import typing

logger = logging.getLogger(__name__)


@dataclass
class SubtitleTrackManager:

    def __init__(
            self,
            file_path: Path,
            options: dict,
    ):
        self.mkv_file = MKVFile(file_path=file_path)
        self.tracks = (track for track in self.mkv_file.tracks if track.track_type == "subtitles" and track.track_codec == "HDMV PGS")
        self.options = options


    def get_pgs_managers(self) -> typing.Generator:
        return (PgsManager(mkv_track=track, mkv_file=self.mkv_file, options=self.options) for track in self.tracks)


