from dataclasses import dataclass
from pathlib import Path
import logging
import typing

from pymkv import MKVFile, MKVTrack

from sub_convert2.pgs.pgs_manager import PgsManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SubtitleTrackManager:
    __slots__ = ("mkv_file", "tracks")

    def __init__(
        self,
        file_path: Path,
    ):
        self.mkv_file = MKVFile(file_path=file_path)
        self.tracks: typing.Iterable[MKVTrack] = (
            track
            for track in self.mkv_file.tracks
            if track.track_type == "subtitles" and track.track_codec == "HDMV PGS"
        )

    def get_pgs_managers(self, options: dict) -> typing.Iterable[PgsManager]:
        return (PgsManager(mkv_track=track, options=options) for track in self.tracks)
