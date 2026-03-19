from subtitle_group import SubtitleGroup, Timeline
from pysrt import SubRipFile, SubRipItem, SubRipTime
from media import PgsSubtitleItem
from dataclasses import dataclass
from collections import Counter
from subtitle_group import Pgs
from langcodes import Language
from itertools import chain
from pymkv import MKVTrack
from colorama import Fore
from pathlib import Path
from PIL import Image
import plotly.express as px
import pandas as pd
import numpy as np
import subprocess
import logging
import hashlib
import shutil


logger = logging.getLogger(__name__)


@dataclass
class PgsManager:
    __slots__ = ("mkv_track", "tmp_path", "pgs", "fallback", "hash")

    def __init__(
        self,
        mkv_track: MKVTrack,
        options: dict,
    ):
        self.mkv_track = mkv_track
        self.hash = hashlib.sha256(str(self.mkv_track).encode()).hexdigest()
        self.tmp_path = Path(f"{options['path_to_tmp']}/{self.hash}")
        self.fallback = options["fallback"] if "fallback" in options else False

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self.tmp_path.mkdir(parents=True)

    def get_pgs_images(self) -> list[tuple[Image.Image, PgsSubtitleItem]]:
        tmp_file = f"{self.tmp_path}/{self.mkv_track.file_path}-{self.mkv_track.track_id}-{self.mkv_track.track_codec}.sup"
        cmd = [
            "mkvextract",
            self.mkv_track.file_path,
            "tracks",
            f"{self.mkv_track.track_id}:{tmp_file}",
        ]

        subprocess.check_output(cmd)
        self.pgs = Pgs(data_reader=open(tmp_file, mode="rb").read)

        pgs_items = self.pgs.items

        for index, item in enumerate(pgs_items):
            image = Image.fromarray(item.image.data)
            image.save(f"tmp/{index}.png")

        self.pgs.dump_display_sets(self.pgs.display_sets)

        shutil.rmtree(path=self.tmp_path)

        return [(Image.fromarray(item.image.data), item) for item in pgs_items]

    def __debug_vis_timelines(self, subtitle_groups: list[SubtitleGroup]):
        df = pd.DataFrame()

        for group in subtitle_groups:
            for entries in group.timelines:
                for timeline in list(chain.from_iterable(entries.values())):
                    tmp = pd.DataFrame(
                        data={
                            "start": [str(timeline.start)],
                            "end": [str(timeline.end)],
                            "placement": [timeline.position],
                            "text": [timeline.text],
                        },
                    )

                    df = pd.concat([df, tmp], ignore_index=True)

        fig = px.timeline(
            data_frame=df,
            x_start="start",
            x_end="end",
            y="placement",
            hover_name="text",
            hover_data="text",
            color="placement",
        )

        Path("debug").mkdir(parents=True, exist_ok=True)
        df.to_json(f"image-test/{self.hash}.json")
        fig.write_image(f"image-test/quickview-{self.hash}.svg")

    def __gen_srt_items(self, subtitle_groups: list[SubtitleGroup]) -> list[SubRipItem]:
        index = 0
        subtitle_items: list[SubRipItem] = []
        for group in subtitle_groups:
            for entries in group.timelines:
                for timeline in list(chain.from_iterable(entries.values())):
                    text = timeline.text
                    start = timeline.start
                    end = timeline.end

                    subtitle_items.append(
                        SubRipItem(index=index, start=start, end=end, text=text)
                    )
                    index += 1

        return subtitle_items

    def save_file(self, format: str = "srt"):

        subtitle_groups: list[SubtitleGroup] = self.pgs.subtitle_groups

        if True:
            self.__debug_vis_timelines(subtitle_groups=subtitle_groups)

        items: list[SubRipItem] = []
        match format:
            case "srt":
                items = self.__gen_srt_items(subtitle_groups=subtitle_groups)
            case "ass":
                raise ValueError(
                    f"Format {format} currently not supported. Support will be added in the future"
                )
            case _:
                raise ValueError(f"Format {format} not valid or supported!")

        srt = SubRipFile(items=items)

        combined: list[list] = []
        if not self.fallback:
            combined: list[list] = savable["combined"]

        track = self.mkv_track

        path = Path(track.file_path).name.replace(".mkv", "")
        counter = Counter()
        average = {}
        weights = {}

        for both in combined:
            for label, prob in both:
                counter.update([label])
                if label not in average:
                    average[label] = [prob]
                else:
                    average[label].append(prob)

        logger.debug(Fore.CYAN + f"{counter}, probablities {average}" + Fore.RESET)

        for label, count in counter.items():
            weights[label] = count / counter.total()
        for label, prob in average.items():
            average[label] = np.average(prob) * weights[label]

        logger.debug(
            Fore.CYAN
            + f"{counter}, probablities {average}, weights: {weights}"
            + Fore.RESET
        )

        final_lang = track.language_ietf
        if not self.fallback:
            final_lang = max(average, key=average.get)

        logger.debug(
            Fore.MAGENTA
            + f"picked language: {final_lang}, averages: {average}"
            + Fore.RESET
        )

        forced = True if track.forced_track or len(items) <= 150 else False

        path = path + ".sdh" if track.flag_hearing_impaired else ""
        path = path + ".forced" if forced else ""
        path = (
            path + "." + track.language
            if track.language_ietf == final_lang
            else Language.get(final_lang).to_alpha3()
            if track.language is not None
            else ""
        )
        potential_path = f"{Path(track.file_path).parent}/{path}.srt"

        logger.debug(
            f"path: {potential_path}, exists prior: {Path(potential_path).exists()}, global: {Path(potential_path).absolute()}"
        )

        logger.debug(
            f"{self.override_if_exists} and {Path(potential_path)} exists: {Path(potential_path).exists()}"
        )
        if self.override_if_exists and Path(potential_path).exists():
            Path(potential_path).unlink()
        else:
            unique = 0
            while Path(potential_path).exists():
                unique += 1
                potential_path = potential_path.replace(
                    f"{path}", f"{path}-{unique}" if unique != 0 else f"{path}"
                )
                potential_path = potential_path.replace(
                    f"-{unique - 1}",
                    "",
                )

        srt.save(path=potential_path)
