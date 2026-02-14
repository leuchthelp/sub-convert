from pgs_manager import PgsManager
from dataclasses import dataclass
from pymkv import MKVFile
from pathlib import Path
import logging
import typing

logger = logging.getLogger(__name__)


@dataclass
class SubtitleTrackManager:
    __slots__ = ("mkv_file", "tracks")

    def __init__(
            self,
            file_path: Path,
    ):
        self.mkv_file = MKVFile(file_path=file_path)
        self.tracks = (track for track in self.mkv_file.tracks if track.track_type == "subtitles" and track.track_codec == "HDMV PGS")


    def get_pgs_managers(self, options: dict) -> typing.Generator:
        return (PgsManager(mkv_track=track, options=options) for track in self.tracks)


