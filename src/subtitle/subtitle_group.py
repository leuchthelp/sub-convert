from dataclasses import dataclass
from itertools import chain
import logging
import typing
import os

from pysrt import SubRipTime
import json

from pgs.pgs_subtitle_item import PgsSubtitleItem, Palette
from pgs.pgs_segments import PgsReader, DisplaySet


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TimelineItem:
    def __init__(
        self,
        start: SubRipTime,
        ds: DisplaySet | None = None,
        end: SubRipTime = SubRipTime(),
        window_id: int = -1,
        index: int = 0,
    ):
        self.start = start
        self.end = end  # will be overwridden by the following TimelineItem item

        if ds is not None:
            self.comp_obj = [
                comp_obj
                for comp_obj in ds.pcs.composition_objects
                if comp_obj.window_id == window_id
            ].pop()

            # Full screen coordiantes for PGS start in the top left; smaller offset = higher up | larger offset = lower down
            self.position = (
                "Top" if self.comp_obj.y_offset < ds.pcs.height / 2 else "Bottom"
            )

            display_obj_cand = [
                display_obj
                for display_obj in ds.ods_segments
                if display_obj.id == self.comp_obj.object_id
            ]
            self.display_obj = display_obj_cand
            self.palette = (
                None if not ds.pds_segments else ds.pds_segments.pop().palettes
            )

        self.index = index
        self.pgs_subtitle_item: PgsSubtitleItem | None
        self.associated_timestamp: list[SubRipTime] = []
        self.overlapping_with: list[TimelineItem] = []
        self.__placeholder: str

    def gen_pgs_subtitle_item(self) -> PgsSubtitleItem:
        if self.display_obj is None or self.palette is None:
            raise ValueError

        self.pgs_subtitle_item = PgsSubtitleItem(
            ods=self.display_obj, comp_obj=self.comp_obj, palette=self.palette
        )
        return self.pgs_subtitle_item

    @property
    def text(self) -> str:
        text: str
        try:
            text = (
                self.pgs_subtitle_item.text
                if self.pgs_subtitle_item is not None
                else self.__placeholder
            )
        except AttributeError:
            text = self.__placeholder
        return text

    def set_text(self, text: str):
        self.__placeholder = text

    @property
    def lang_estimate(self) -> list[tuple[str, typing.Any]]:
        tmp: list[tuple[str, typing.Any]] = []
        try:
            tmp = (
                self.pgs_subtitle_item.lang_estimate
                if self.pgs_subtitle_item is not None
                else []
            )
        except AttributeError:
            pass
        return tmp

    @property
    def duration(self) -> SubRipTime:
        if self.end is None:
            raise ValueError("End has not been set yet.")
        return self.end - self.start

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self}]>"

    def __str__(self):
        return f"[{self.start} --> {self.end or ''}]"


@dataclass
class SubtitleGroup:
    __slots__ = ("pgs_subtitle_items", "timelines", "overlap")

    def __init__(
        self,
        members: list[DisplaySet],
    ):
        self.overlap = self.__find_overlap(members=members)

        end = members[-1]
        global_palettes: dict[int, list[Palette]] = {}
        timelines: list[dict[str, list[TimelineItem]]] = []

        if self.overlap:
            global_palettes = self.__find_global_palettes(members=members)
            redef_positions = self.__find_redefintion_positions(members=members)
            reset_positions = self.__find_reset_positions(members=members)
            overlapping = self.__find_overlapping(
                reset_positions=reset_positions,
                redef_positions=redef_positions,
                members=members,
            )

            tmp = [
                self.__gen_timelines(members=segment, global_palettes=global_palettes)
                for _, segment in overlapping.items()
            ]
            reset_statements = [members[index] for index in reset_positions]
            redef_statements = [members[index] for index in redef_positions]

            for index, fixables in enumerate(tmp):
                actual_end = (
                    redef_statements[index + 1] if index + 1 < len(tmp) else end
                )
                timelines.append(
                    self.__fix_endpoints(
                        fixables=fixables,
                        reset_statements=reset_statements[index],
                        end=actual_end,
                    )
                )
        else:
            tmp = self.__gen_timelines(members=members)
            timelines.append(
                self.__fix_endpoints(fixables=tmp, reset_statements=end, end=end)
            )

        self.timelines = timelines
        self.pgs_subtitle_items = self.__gen_pgs_subtitle_items(
            timelines=self.timelines
        )

    def __find_overlap(self, members: list[DisplaySet]) -> bool:
        for ds in members:
            if ds.pcs.number_composition_objects > 1:
                return True
        return False

    def __find_global_palettes(
        self, members: list[DisplaySet]
    ) -> dict[int, list[Palette]]:
        """
        Grab all Palette defined at either EPOCH_START, ACQUISITION_POINT or intermediate with varying IDs.

        Returns
        -------

        list
            Contains all Palettes found in the global Palette definition at ACQUISITION_POINT or intermediate with varying IDs
        """
        global_palettes: dict[int, list[Palette]] = {}
        for member in members:
            for pds_segment in member.pds_segments:
                if pds_segment.palette_id not in global_palettes:
                    global_palettes[pds_segment.palette_id] = pds_segment.palettes
        return global_palettes

    def __find_reset_positions(self, members: list[DisplaySet]) -> list[int]:
        """
        In PGS Files END segments are usually sized to 11 bytes, contain no objects & are placed at the end of the group.
        However, if elements overlap an additional intermediate RESET segment is inserted which drops the number of objects
        and marks the position a Palette update can happen and either new Elements can overlap or the overlap ends.

        PGS subtitles can only show 2 objects on screen at once.

        This finds these positions.

        They seem to always be marked with a size of 19 bytes and dropping the number of segments, which will always be 1.

        Returns
        -------

        list
            Contains the indices of the RESET segments
        """
        reset_positions = []
        for index, ds in enumerate(members):
            if (
                ds.pcs.size == 19
                and ds.pcs.number_composition_objects == 1
                and not ds.ods_segments
            ):
                reset_positions.append(index)
        return reset_positions

    def __find_redefintion_positions(self, members: list[DisplaySet]) -> list[int]:
        redef_positions = []
        for index, ds in enumerate(members):
            if (
                ds.pcs.size == 19
                and ds.pcs.number_composition_objects == 1
                and ds.ods_segments
                and ds.pds_segments
            ):
                redef_positions.append(index)
        return redef_positions

    def __find_overlapping(
        self,
        reset_positions: list[int],
        redef_positions: list[int],
        members: list[DisplaySet],
    ) -> dict[int, list[DisplaySet]]:
        overlapping: dict[int, list[DisplaySet]] = {}

        for pos in range(len(reset_positions)):
            start = redef_positions[pos]
            stop = reset_positions[pos]
            overlapping[start] = members[start : stop + 1]
        return overlapping

    def __gen_timelines(
        self, members: list[DisplaySet], global_palettes: dict[int, list[Palette]] = {}
    ) -> dict[str, list[TimelineItem]]:
        timelines: dict[str, list[TimelineItem]] = {}

        for ds in members:
            for comp_obj in ds.pcs.composition_objects:
                for window in ds.wds.windows:
                    if window.window_id == comp_obj.window_id:
                        new_timeline = TimelineItem(
                            window_id=comp_obj.window_id,
                            start=ds.pcs.presentation_timestamp,
                            ds=ds,
                        )

                        if new_timeline.position in timelines:
                            prev_timeline = timelines[new_timeline.position][-1]
                            new_timeline.index = prev_timeline.index + 1

                            prev_timeline.end = new_timeline.start
                            prev_timeline.associated_timestamp.append(prev_timeline.end)

                            if (
                                new_timeline.comp_obj.object_id
                                != prev_timeline.comp_obj.object_id
                            ):
                                if not new_timeline.palette:
                                    new_timeline.palette = prev_timeline.palette

                                if not new_timeline.display_obj:
                                    new_timeline.display_obj = prev_timeline.display_obj

                                timelines[new_timeline.position].append(new_timeline)
                        else:
                            if not new_timeline.palette:
                                new_timeline.palette = global_palettes[
                                    ds.pcs.palette_id
                                ]
                            new_timeline.associated_timestamp.append(new_timeline.start)
                            timelines[new_timeline.position] = [new_timeline]
        return timelines

    def __fix_endpoints(
        self,
        fixables: dict[str, list[TimelineItem]],
        reset_statements: DisplaySet,
        end: DisplaySet,
    ) -> dict[str, list[TimelineItem]]:
        for _, items in fixables.items():
            fixable = items[-1]

            if (
                reset_statements.pcs.composition_objects
                and fixable.comp_obj.object_id
                != reset_statements.pcs.composition_objects[0].object_id
            ):
                fixable.end = reset_statements.pcs.presentation_timestamp
                fixable.associated_timestamp.append(fixable.end)
            else:
                fixable.end = end.pcs.presentation_timestamp
                fixable.associated_timestamp.append(fixable.end)

        return fixables

    def __gen_pgs_subtitle_items(
        self, timelines: list[dict[str, list[TimelineItem]]]
    ) -> list[PgsSubtitleItem]:
        items: list[PgsSubtitleItem] = []

        for timeline in timelines:
            for _, entries in timeline.items():
                for element in entries:
                    items.append(element.gen_pgs_subtitle_item())
        return items


class Pgs:
    __slots__ = (
        "data_reader",
        "temp_folder",
        "_items",
        "display_sets",
        "subtitle_groups",
    )

    def __init__(
        self,
        data_reader: typing.Callable[[], bytes],
        temp_folder="tmp",
    ):
        self.data_reader = data_reader
        self.temp_folder = temp_folder
        self._items: typing.Optional[typing.List[PgsSubtitleItem]] = None

    @property
    def items(self) -> list[PgsSubtitleItem]:
        if self._items is None:
            data = self.data_reader()
            self._items = self.__decode(data)
        return self._items

    def __decode(self, data: bytes) -> list[PgsSubtitleItem]:
        display_sets = list(PgsReader.decode(data))
        groups: typing.List[typing.List[DisplaySet]] = []
        self.display_sets = display_sets

        tmp = []
        for ds in display_sets:
            if ds.is_start() or (
                ds.is_normal() and len(ds.ods_segments) != 0 or ds.pcs.size == 19
            ):
                tmp.append(ds)
            elif (
                len(ds.ods_segments) == 0
                and len(ds.pds_segments) == 0
                and ds.is_normal()
                and ds.pcs.size == 11
            ):
                tmp.append(ds)
                groups.append(tmp)
                tmp = []

        # test_groups = list(range(100, 112))
        # sliced = [ds for group in groups for ds in group if ds.index in test_groups]
        # self.subtitle_groups = [SubtitleGroup(members=sliced)]

        self.subtitle_groups = [SubtitleGroup(members=group) for group in groups]

        return list(
            chain.from_iterable(
                [group.pgs_subtitle_items for group in self.subtitle_groups]
            )
        )

    def dump_display_sets(self, display_sets: typing.List[DisplaySet], path=""):
        new_line = "\n"

        actual_path = self.temp_folder if not path else path

        with open(
            os.path.join(actual_path, "display-sets.txt"),
            mode="w",
            encoding="utf8",
        ) as f:
            f.write(f"{new_line.join([str(ds) for ds in display_sets])}")

        with open(
            os.path.join(actual_path, "display-sets.json"),
            mode="w",
            encoding="utf8",
        ) as f:
            json.dump(
                [ds.to_json() for ds in display_sets],
                f,
                indent=2,
                ensure_ascii=False,
                default=lambda x: str(x),
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self}]>"

    def __enter__(self):
        return self
