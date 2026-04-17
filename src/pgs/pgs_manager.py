from dataclasses import dataclass
from collections import Counter
from itertools import chain
from pathlib import Path
import subprocess
import logging
import hashlib
import typing
import shutil

from pysrt import SubRipFile, SubRipItem, SubRipTime
from PIL import Image, ImageOps
from langcodes import Language
from pymkv import MKVTrack
import numpy as np

from subtitle.subtitle_group import SubtitleGroup, TimelineItem, Pgs
from pgs.pgs_subtitle_item import PgsSubtitleItem


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_between(start: SubRipTime, end: SubRipTime, now: SubRipTime) -> bool:
    between = False
    between |= start <= now <= end
    between |= end < start and (start <= now or now <= end)
    return between


@dataclass
class PgsManager:
    __slots__ = (
        "mkv_track",
        "tmp_path",
        "pgs",
        "hash",
        "overwrite_if_exists",
        "dump_debug",
    )

    def __init__(
        self,
        mkv_track: MKVTrack,
        options: dict,
    ):
        self.mkv_track = mkv_track
        self.hash = hashlib.sha256(str(self.mkv_track).encode()).hexdigest()
        self.tmp_path = Path(f"{options['path_to_tmp']}/{self.hash}")
        self.overwrite_if_exists = (
            options["overwrite_if_exists"]
            if "overwrite_if_exists" in options
            else False
        )
        self.dump_debug = options["dump_debug"] if "dump_debug" in options else False

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self.tmp_path.mkdir(parents=True)

        self.pgs = None

    def get_pgs_images(self) -> list[tuple[Image.Image, PgsSubtitleItem]]:
        tmp_file = (
            f"{self.tmp_path}/{self.mkv_track.file_path}"
            + f"-{self.mkv_track.track_id}-{self.mkv_track.track_codec}.sup"
        )
        cmd = [
            "mkvextract",
            self.mkv_track.file_path,
            "tracks",
            f"{self.mkv_track.track_id}:{tmp_file}",
        ]

        subprocess.check_output(cmd)
        self.pgs = Pgs(tmp_location=tmp_file)

        pgs_items = self.pgs.items

        final: list[tuple[Image.Image, PgsSubtitleItem]] = []
        for item in pgs_items:
            # Expand border to ensure proper recognition if text is very close to image borders.
            # Also invert as black-outline texts is saved inverted (as white-outline).
            # This could help detection.
            image = Image.fromarray(item.image.data).convert("RGB")
            rgb = image.getpixel((0, 0))

            color = "Black" if rgb == (0, 0, 0) else "White"
            image = ImageOps.expand(image=image, border=10, fill=color)
            image = ImageOps.invert(image)
            final.append((image, item))

        if self.dump_debug:
            debug_path = Path("debug")
            debug_path.mkdir(parents=True, exist_ok=True)
            path = Path(
                (
                    f"{debug_path}/{self.hash[0:6]}"
                    + f"-{Path(self.mkv_track.file_path).name}"
                    + f"-{self.mkv_track.track_id}"
                )
            )

            image_path = Path(f"{path}/images")
            image_path.mkdir(parents=True, exist_ok=True)
            for index, (image, item) in enumerate(final):
                image.save(f"{image_path.absolute()}/{index}.png")
            
            if self.pgs.display_sets is not None:
                self.pgs.dump_display_sets(self.pgs.display_sets, path=str(path.absolute()))

        shutil.rmtree(path=self.tmp_path)
        return final

    def __debug_vis_timelines(self, subtitle_groups: list[SubtitleGroup]):
        from datetime import datetime

        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame()

        for group in subtitle_groups:
            for timeline in group.timelines:
                for item in list(chain.from_iterable(timeline.values())):
                    formating = "%H:%M:%S,%f"
                    tmp = pd.DataFrame(
                        data={
                            "start": [datetime.strptime(str(item.start), formating)],
                            "end": [datetime.strptime(str(item.end), formating)],
                            "placement": [item.position],
                            "text": [item.text],
                        },
                    )

                    df = pd.concat([df, tmp], ignore_index=True)

        # pio.get_chrome()
        fig = px.timeline(
            data_frame=df,
            x_start="start",
            x_end="end",
            y="placement",
            hover_name="text",
            hover_data="text",
            color="placement",
            color_discrete_map={
                "Top": "#AB029E",
                "Bottom": "#FF4430",
            },
        )

        debug_path = Path("debug")
        debug_path.mkdir(parents=True, exist_ok=True)

        path = Path(
            (
                f"{debug_path}/{self.hash[0:6]}"
                + f"-{Path(self.mkv_track.file_path).name}"
                + f"-{self.mkv_track.track_id}"
            )
        )
        path.mkdir(parents=True, exist_ok=True)
        df.to_json(f"{path.absolute()}/{self.hash[0:6]}.json")
        fig.write_image(f"{path.absolute()}/quickview-{self.hash[0:6]}.svg")

    def __timeline_events(self, timeline: dict[str, list[TimelineItem]]):
        tmp = list(
            chain.from_iterable(
                [
                    (item.start, item.end)
                    for item in chain.from_iterable(timeline.values())
                ]
            )
        )

        timeline_events: list[SubRipTime] = []
        for x in tmp:
            if x not in timeline_events:
                timeline_events.append(x)
        timeline_events.sort()

        return timeline_events

    def __sweeping_line(self, timeline: dict[str, list[TimelineItem]]):
        timeline_events = self.__timeline_events(timeline=timeline)

        intermediate: list[TimelineItem] = []
        timeline_items = list(chain.from_iterable(timeline.values()))
        for index, event in enumerate(timeline_events):
            overlapping: list[TimelineItem] = []

            for item in timeline_items:
                if (
                    item.start == event
                    or item.end != event
                    and is_between(item.start, item.end, event)
                ):
                    overlapping.append(item)

            if len(overlapping) == 1:
                item = overlapping[0]

                start = event
                end: SubRipTime
                try:
                    end = timeline_events[index + 1]
                except IndexError:
                    end = item.end

                new_tline = item
                if item.start != start or item.end != end:
                    new_tline = TimelineItem(start=start, end=end)
                    new_tline.set_text(item.text)

                intermediate.append(new_tline)

            if len(overlapping) == 2:
                bottom, top = (
                    (overlapping[0], overlapping[1])
                    if overlapping[0].position == "Bottom"
                    else (overlapping[1], overlapping[0])
                )

                start = event
                end: SubRipTime
                try:
                    end = timeline_events[index + 1]
                except IndexError:
                    end = max(bottom.end, top.end)

                new_tline = TimelineItem(start=start, end=end)
                new_tline.set_text(top.text + "\n-\n" + bottom.text)
                intermediate.append(new_tline)

        return intermediate

    def __srt_combine_timelines(
        self, subtitle_groups: list[SubtitleGroup]
    ) -> list[TimelineItem]:
        intermediate: list[TimelineItem] = []

        for group in subtitle_groups:
            for timeline in group.timelines:
                if group.overlap:
                    intermediate = intermediate + self.__sweeping_line(
                        timeline=timeline
                    )
                else:
                    for item in list(chain.from_iterable(timeline.values())):
                        intermediate.append(item)

        return intermediate

    def __gen_srt_items(self, subtitle_groups: list[SubtitleGroup]) -> list[SubRipItem]:
        intermediate = self.__srt_combine_timelines(subtitle_groups=subtitle_groups)

        subtitle_items: list[SubRipItem] = []
        for index, item in enumerate(intermediate):
            subtitle_items.append(
                SubRipItem(index=index, start=item.start, end=item.end, text=item.text)
            )
        return subtitle_items

    def __get_lang_weights(
        self, subtitle_groups: list[SubtitleGroup]
    ) -> dict[str, list]:
        combined: list[list[tuple[str, typing.Any]]] = []
        for group in subtitle_groups:
            for timeline in group.timelines:
                for item in list(chain.from_iterable(timeline.values())):
                    combined.append(item.lang_estimate)

        counter = Counter()
        average: dict[str, list] = {}
        weights = {}

        for both in combined:
            for label, prob in both:
                counter.update([label])
                if label not in average:
                    average[label] = [prob]
                else:
                    average[label].append(prob)

        for label, count in counter.items():
            weights[label] = count / counter.total()
        for label, prob in average.items():
            average[label] = np.average(prob) * weights[label]

        return average

    def save_file(self, export_as: str = "srt"):

        subtitle_groups: list[SubtitleGroup] = []
        if self.pgs is not None:
            subtitle_groups = self.pgs.subtitle_groups

        if self.dump_debug:
            self.__debug_vis_timelines(subtitle_groups=subtitle_groups)

        items: list[SubRipItem] = []
        match export_as:
            case "srt":
                items = self.__gen_srt_items(subtitle_groups=subtitle_groups)
            case "ass":
                raise ValueError(
                    (
                        f"Format {export_as} currently not supported. "
                        + "Support will be added in the future"
                    )
                )
            case _:
                raise ValueError(f"Format {export_as} not valid or supported!")

        srt = SubRipFile(items=items)
        track = self.mkv_track
        path = Path(track.file_path).name.replace(".mkv", "")
        average = self.__get_lang_weights(subtitle_groups=subtitle_groups)

        final_lang = track.language_ietf if track.language_ietf is not None else ""
        final_lang = max(average, key=average.get)  # type: ignore

        forced = bool(track.forced_track or len(items) <= 150)
        path = path + ".sdh" if track.flag_hearing_impaired else ""
        path = path + ".forced" if forced else ""

        if track.language_ietf == final_lang:
            if track.language is not None:
                path = path + "." + track.language
        else:
            path = path + "." + Language.get(final_lang).to_alpha3()

        potential_path = f"{str(Path(track.file_path)).replace('.mkv', '')}{path}.srt"
        if self.overwrite_if_exists and Path(potential_path).exists():
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
