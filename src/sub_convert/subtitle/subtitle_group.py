from dataclasses import dataclass
from collections import Counter
import logging
import typing
import json
import os

from ..pgs.pgs_segments import PgsReader, DisplaySet
from ..pgs.pgs_subtitle_item import PgsSubtitleItem, Palette
from ..subtitle.timeline import (
    TimelineItem,
    look_to_combine,
    gen_timelines,
    fix_endpoints,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class FadeHandler:
    def __init__(
        self, members: list[DisplaySet], global_palettes: dict[int, list[list[Palette]]]
    ):
        timelines: list[dict[str, list[TimelineItem]]] = []

        # Check members for fade, repeat until all members have been checked
        fade_groups, fade_pos = self.__find_fade(members)

        if fade_pos:
            for idx, group in enumerate(fade_groups):
                if idx in fade_pos:
                    timelines.append(self.__handle_fade(group))
                else:
                    tmp = gen_timelines(group[:-1], global_palettes)
                    tmp = fix_endpoints(tmp, group[-1], group[-1])
                    timelines.append(tmp)

        self.timelines = timelines

    def __find_roi(self, members: list[DisplaySet]):
        rois: list[list[DisplaySet]] = []
        check, start, end = members[1:-1], members[0], members[-1]
        # A region of interest (roi) in the context of a fade (in / out)
        # lies between 2 segments which only consist of ACQUISITION_POINTS
        # and a START or END segment.
        # Within a fade the same image is displayed with varying levels of
        # ALPHA, which causes the image to be identical in either height and
        # width.
        # Finding these regions should only require looking at the relative
        # size of the ObjectDefinitionSegment and grouping them as such.
        tmp: list[DisplaySet] = []
        tmp.append(start)
        for ds in check:
            if not ds.is_acquisition_point():
                tmp.append(ds)
                rois.append(tmp)
                tmp = [ds]
                continue

            curr = ds.ods_segments[-1]
            lead = tmp[0].ods_segments[-1]

            height_delta = abs(curr.height - lead.height)
            width_delta = abs(curr.width - lead.width)

            if (
                (curr.version == 0 or lead.version != curr.version)
                and height_delta <= 2
                and width_delta <= 2
            ):
                tmp.append(ds)
            else:
                tmp.append(ds)
                rois.append(tmp)
                tmp = [ds]

        if tmp:
            rois.append(tmp)
        rois[-1].append(end)
        return rois

    def __find_fade(self, members: list[DisplaySet]):
        fade_groups: list[list[DisplaySet]] = []
        fade_pos: list[int] = []
        fade_groups = self.__find_roi(members)

        # Filter rios; within rois there might still not be a fade
        # as such we need to filter them, based on the amount of time
        # each DisplayObject is being shown. If it is being shown for
        # less than half a second, most likely a fade is taking place
        for idx, group in enumerate(fade_groups):
            start = group[0].pcs.presentation_timestamp
            end = group[-1].pcs.presentation_timestamp

            duration = end - start
            converted = float(f"{duration.seconds}.{duration.milliseconds}")
            delta = converted / len(group)

            if not (len(group) <= 2 or delta > 0.5):
                fade_pos.append(idx)
                continue

        return fade_groups, fade_pos

    def __select_best_image(
        self, group: list[DisplaySet], item: TimelineItem, guess="IN"
    ) -> TimelineItem:

        selected: DisplaySet
        match guess.lower():
            case "in":
                selected = group[-2]
            case "out":
                selected = group[0]
            case _:
                selected = group[len(group) // 2]

        item.display_obj = selected.ods_segments
        if selected.pds_segments:
            item.palette = selected.pds_segments[0].palettes
        else:
            item.palette

        return item

    def __guess_fade_in_out(
        self, group: list[DisplaySet], item: TimelineItem
    ) -> TimelineItem:
        guess = "Unknown"

        start = group[0].ods_segments[-1]
        end = group[-2].ods_segments[-1]

        if start.data_len > end.data_len:
            guess = "OUT"

        if end.data_len > start.data_len:
            guess = "IN"

        item = self.__select_best_image(group, item, guess)
        return item

    def __handle_fade(self, group: list[DisplaySet]):
        comp_obj = group[0].pcs.composition_objects[-1]
        window = group[0].wds.windows[-1]
        t_start = group[0].pcs.presentation_timestamp
        t_end = group[-1].pcs.presentation_timestamp
        new_item = TimelineItem(t_start, comp_obj, window, group[0], t_end)

        new_item = self.__guess_fade_in_out(group, new_item)

        new_timeline = {new_item.position: [new_item]}
        return new_timeline


@dataclass
class SubtitleGroup:
    """
    Defines an instance of a SubtitleGroup. A SubtitleGroup wraps around N DisplaySets
    usually defining a START segment with the following segments defining more objects
    to display until an END segment.

    Within a group subtitles can overlap as PGS supports two windows displaying one image
    each at a time. The current image is displayed until the window is updated with
    another image.

    Parameters
    ----------
    members: list
        List of DisplaySets the SubtitleGroup wraps around.
    """

    __slots__ = ("pgs_subtitle_items", "timelines", "overlap", "occurrences")

    def __init__(
        self,
        members: list[DisplaySet],
    ):
        self.overlap = self.__find_overlap(members=members)

        end = members[-1]
        global_palettes: dict[int, list[list[Palette]]] = {}
        global_palettes = self.__find_global_palettes(members=members)

        timelines: list[dict[str, list[TimelineItem]]] = []

        if self.overlap:
            reset_positions = self.__find_reset_positions(members=members)
            redef_positions = self.__find_redefinition_positions(
                members=members, reset_pos=reset_positions
            )

            overlapping, reset_positions, redef_positions = self.__define_overlapping(
                members, reset_positions, redef_positions
            )

            tmp = [
                gen_timelines(members=segment, global_palettes=global_palettes)
                for _, segment in overlapping.items()
            ]
            reset_statements = [members[index] for index in reset_positions]
            redef_statements = [members[index] for index in redef_positions]

            for index, fixables in enumerate(tmp):
                actual_end = (
                    redef_statements[index + 1] if index + 1 < len(tmp) else end
                )
                timelines.append(
                    fix_endpoints(
                        fixables=fixables,
                        reset_statements=reset_statements[index],
                        end=actual_end,
                    )
                )

            timelines = look_to_combine(timelines=timelines)
        else:
            fade_hint = len(members) > 5

            if fade_hint:
                fade_handler = FadeHandler(members, global_palettes)
                timelines = timelines + fade_handler.timelines

            if not timelines:
                for idx, ds in enumerate(members):
                    if ds is end:
                        break

                    tmp = gen_timelines([ds], global_palettes)
                    tmp = fix_endpoints(tmp, members[idx + 1], end)
                    timelines.append(tmp)

        self.timelines = timelines
        self.pgs_subtitle_items = self.__gen_pgs_subtitle_items(
            timelines=self.timelines
        )

    def __find_overlap(self, members: list[DisplaySet]) -> bool:
        """
        Check if the current set of DisplaySets between a EPOCH_START, ACQUISITION_POINT &
        EndSegment have overlapping Windows. In PGS there can be at most two overlapping
        Windows at a time.

        Returns
        -------

        bool
            Returns TRUE right away if there is any overlap found.
        """
        for ds in members:
            if ds.pcs.number_composition_objects > 1:
                return True
        return False

    def __acquisition_point_present(self, members: list[DisplaySet]) -> int:
        for index, ds in enumerate(members):
            if ds.pcs.is_acquisition_point():
                return index

        return -1

    def __find_global_palettes(
        self, members: list[DisplaySet]
    ) -> dict[int, list[list[Palette]]]:
        """
        Grab all Palette defined at either EPOCH_START, ACQUISITION_POINT or intermediate with
        varying IDs.

        Returns
        -------

        list
            Contains all Palettes found in the global Palette definition at ACQUISITION_POINT
            or intermediate with varying IDs.
        """
        global_palettes: dict[int, list[list[Palette]]] = {}
        for ds in members:
            for pds_segment in ds.pds_segments:
                if pds_segment.palette_id not in global_palettes:
                    global_palettes[pds_segment.palette_id] = [pds_segment.palettes]
                else:
                    global_palettes[pds_segment.palette_id].append(pds_segment.palettes)
        return global_palettes

    def __find_reset_positions(self, members: list[DisplaySet]) -> list[int]:
        """
        In PGS files END segments are usually sized to 11 bytes, contain no objects
        & are placed at the end of the group. However, if elements overlap an additional
        intermediate RESET segment is inserted which drops the number of objects and marks the
        position a Palette update can happen & either new elements can overlap or the overlap ends.

        PGS subtitles can only show 2 objects on screen at once.

        This finds these positions.

        They seem to always be marked with a size of 19 bytes and dropping the number of segments,
        which will always be 1.

        Returns
        -------

        list
            Contains the indices of the RESET segments.
        """
        reset_positions = []
        for index, ds in enumerate(members):
            if (
                ds.pcs.size == 19
                and ds.pcs.number_composition_objects == 1
                and not ds.ods_segments
            ):
                reset_positions.append(index)

            if ds.pcs.is_acquisition_point():
                reset_positions.append(index)

        return reset_positions

    def __find_redefinition_positions(
        self, members: list[DisplaySet], reset_pos: list[int]
    ) -> list[int]:
        """
        In PGS REDEF segments usually define a new set of Palettes, Windows and CompositionObjects.
        They also define the number of Windows currently active. REDEF segments usually follow RESET
        segments as they define new content and their positions about to come after.

        This finds these positions.

        A START segment is also a valid REDEF segment.

        Returns
        -------

        list
            Contains indices of REDEF segments.
        """
        redef_positions = []
        for index, ds in enumerate(members):
            if (
                ds.pcs.size in [19, 27]
                and ds.pcs.number_composition_objects in [1, 2]
                and ds.ods_segments
                # and ds.pds_segments
            ):
                if (
                    index - 1 in reset_pos
                    or ds.pcs.is_start()
                    or ds.pcs.is_acquisition_point()
                ):
                    redef_positions.append(index)

        return redef_positions

    def __find_overlapping(
        self,
        reset_positions: list[int],
        redef_positions: list[int],
        members: list[DisplaySet],
    ) -> dict[int, list[DisplaySet]]:
        """
        Finds the actual DisplaySets which are overlapping starting from each REDEF segment position
        until the immediately following RESET segment is reached.

        Returns
        -------

        dict
            Contains all DisplaySets that are overlapping grouped by the REDEF segments position.
        """
        overlapping: dict[int, list[DisplaySet]] = {}

        for pos, reset in enumerate(reset_positions):
            start = redef_positions[pos]
            stop = reset
            overlapping[start] = members[start : stop + 1]

        return overlapping

    def __define_overlapping(
        self, members: list[DisplaySet], reset_pos: list[int], redef_pos: list[int]
    ):
        acquisition_point_present = self.__acquisition_point_present(members=members)
        if acquisition_point_present != -1:
            new_reset: list[int] = []
            new_redef: list[int] = []

            if members[acquisition_point_present - 1].pcs.is_start():
                new_reset.append(acquisition_point_present)
                new_redef.append(acquisition_point_present - 1)

            if acquisition_point_present - 1 in reset_pos:
                new_reset.append(acquisition_point_present - 1)
                new_redef.append(0)

            new_reset.append(reset_pos[-1])
            new_redef.append(acquisition_point_present)

            reset_pos = new_reset
            redef_pos = new_redef

        overlapping = self.__find_overlapping(
            reset_positions=reset_pos,
            redef_positions=redef_pos,
            members=members,
        )

        return overlapping, reset_pos, redef_pos

    def __gen_pgs_subtitle_items(
        self, timelines: list[dict[str, list[TimelineItem]]]
    ) -> list[PgsSubtitleItem]:
        """
        Generate PgsSubtitleItems which hold metadata on the image as a PgsImage converted from
        raw bytes and the matching Palette defined.

        Returns
        -------

        list
            List containing PgsSubtitleItems which will eventual contain the text extracted from
            the PGS image.
        """
        self.occurrences = Counter()
        items: list[PgsSubtitleItem] = []

        for timeline in timelines:
            for _, entries in timeline.items():
                for element in entries:
                    self.occurrences.update((element.position,))
                    items.append(element.gen_pgs_subtitle_item())
        return items


@dataclass
class Pgs:
    """
    An instance of PGS represent a mapping of a PGS file to Python.
    The PGS.items property contains all PgsSubtitleItem contained within
    the specified PGS file.

    Parameters
    ----------
    tmp_location: str
        Location of a prior extract .sub PGS file.
    temp_folder: str
        Only necessary for debugging. Directory where to dump the metadata. Defaults to \"tmp\"
    """

    __slots__ = (
        "tmp_location",
        "temp_folder",
        "_items",
        "subtitle_groups",
        "occurrences",
    )

    def __init__(
        self,
        tmp_location: str,
        temp_folder="tmp",
    ):
        self.tmp_location = tmp_location
        self.temp_folder = temp_folder
        self._items: typing.Optional[list[PgsSubtitleItem]] = None
        self.occurrences: Counter[str] = Counter()

    @property
    def items(self) -> list[PgsSubtitleItem]:
        """
        Return PgsSubtitleItems which hold metadata on the image as a PgsImage converted from
        raw bytes.

        Returns
        -------

        list
            List containing PgsSubtitleItems which will eventual contain the text extracted from
            the PGS image.
        """
        if self._items is None:
            with open(self.tmp_location, "+rb") as data:
                self._items = self.__decode(data.read())
        return self._items

    def __decode(self, data: bytes) -> list[PgsSubtitleItem]:
        """
        Decodes the PGS file provided as raw bytes and group the contained
        DisplaySets into unique PgsSubtitleItems.

        Returns
        -------

        list
            List containing PgsSubtitleItems which will eventual contain the text extracted from
            the PGS image.
        """
        display_sets = list(PgsReader.decode(data))

        if self.temp_folder != "tmp":
            self.dump_display_sets(display_sets=display_sets)

        groups: list[list[DisplaySet]] = []
        tmp = []
        for ds in display_sets:
            if (
                ds.is_start()
                or ds.is_acquisition_point()
                or (ds.is_normal() and len(ds.ods_segments) != 0 or ds.pcs.size == 19)
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

        # Debug helper code
        # test_groups = list(range(82, 131))
        # test_groups = list(range(131, 159))
        # test_groups = list(range(315, 323))
        # sliced = [ds for group in groups for ds in group if ds.index in test_groups]
        # self.subtitle_groups = [SubtitleGroup(members=sliced)]

        self.subtitle_groups = [SubtitleGroup(members=group) for group in groups]
        res: list[PgsSubtitleItem] = []
        for group in self.subtitle_groups:
            self.occurrences.update(group.occurrences)
            res += group.pgs_subtitle_items

        return res

    def dump_display_sets(self, display_sets: list[DisplaySet], path=""):
        """
        Dumps DisplaySets contained in PGS file as .txt and .json
        """
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
                default=str,
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self}]>"

    def __enter__(self):
        return self
