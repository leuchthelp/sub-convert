from dataclasses import dataclass
from itertools import chain
import logging
import typing
import json
import os

from pysrt import SubRipTime

from pgs.pgs_subtitle_item import PgsSubtitleItem, Palette
from pgs.pgs_segments import PgsReader, DisplaySet


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TimelineItem:
    """
    An instance of TimelineItem describes an objects being displayed either
    a Top or Bottom timeline within a PGS file.

    A TimelineItem is effectively the text block being displayed on screen
    for a set duration.

    Parameters
    ----------
    start: SubRipTime
        When the item starts being displayed.
    ds: DisplaySet
        DisplaySet associated with this item.
    end: SubRipTime
        When the item stops being displayed.
    window_id: int
        The Window the item is being displayed in within a PGS file.
    """

    def __init__(
        self,
        start: SubRipTime,
        ds: DisplaySet | None = None,
        end: SubRipTime = SubRipTime(),
        window_id: int = -1,
    ):
        self.start = start
        self.end = end  # will be overwridden by the following TimelineItem item

        if ds is not None:
            self.comp_obj = [
                comp_obj
                for comp_obj in ds.pcs.composition_objects
                if comp_obj.window_id == window_id
            ].pop()

            # Full screen coordiantes for PGS start in the top left;
            # smaller offset = higher up | larger offset = lower down
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

        self.pgs_subtitle_item: PgsSubtitleItem | None
        self.overlapping_with: list[SubRipTime] = []
        self.__placeholder: str

    def gen_pgs_subtitle_item(self) -> PgsSubtitleItem:
        """
        Generates a PgsSubtitleItem described by the TimelineItems entry.
        Contains the image and later text / language estimation of the text.

        Returns
        -------

        PgsSubtitleItem
            The PgsSubtitleItem which is displayed within this timeline slot.
        """
        if self.display_obj is None or self.palette is None:
            raise ValueError

        self.pgs_subtitle_item = PgsSubtitleItem(
            ods=self.display_obj, comp_obj=self.comp_obj, palette=self.palette
        )
        return self.pgs_subtitle_item

    @property
    def text(self) -> str:
        """
        Returns text displayed within this timeline slot in a PGS file.

        Returns
        -------

        str
            Text displayed within this timeline slot in a PGS file.
        """
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
        """
        Sets the text displayed within this timeline slot in a PGS file.
        """
        self.__placeholder = text

    @property
    def lang_estimate(self) -> list[tuple[str, typing.Any]]:
        """
        Contains a list of languages and their probabilities matching
        the text within a PgsSubtitleItem.

        Returns
        -------

        list
            Language estimation of the text.
        """
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
        """
        Provides duration with which a given TimelineItem is being displayed.

        Returns
        -------

        SubRipTime
            Duration with which a given TimelineItem is being displayed.
        """
        if self.end is None:
            raise ValueError("End has not been set yet.")
        return self.end - self.start

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self}]>"

    def __str__(self):
        return f"[{self.start} --> {self.end or ''}]"


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

    __slots__ = ("pgs_subtitle_items", "timelines", "overlap")

    def __init__(
        self,
        members: list[DisplaySet],
    ):
        self.overlap = self.__find_overlap(members=members)

        end = members[-1]
        global_palettes: dict[int, list[Palette]] = {}
        global_palettes = self.__find_global_palettes(members=members)

        timelines: list[dict[str, list[TimelineItem]]] = []

        if self.overlap:
            reset_positions = self.__find_reset_positions(members=members)
            redef_positions = self.__find_redefintion_positions(
                members=members, reset_pos=reset_positions
            )

            acquisition_point_present = self.__acquisition_point_present(
                members=members
            )
            if acquisition_point_present != -1:
                new_reset: list[int] = []
                new_redef: list[int] = []

                if members[acquisition_point_present - 1].pcs.is_start():
                    new_reset.append(acquisition_point_present)
                    new_redef.append(acquisition_point_present - 1)

                new_reset.append(reset_positions[-1])
                new_redef.append(acquisition_point_present)

                reset_positions = new_reset
                redef_positions = new_redef

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

            timelines = self.__look_to_combine(timelines=timelines)
        else:
            tmp = self.__gen_timelines(members=members, global_palettes=global_palettes)
            timelines.append(
                self.__fix_endpoints(fixables=tmp, reset_statements=end, end=end)
            )

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
            if ds.pcs.is_aquisition_point():
                return index

        return -1

    def __find_global_palettes(
        self, members: list[DisplaySet]
    ) -> dict[int, list[Palette]]:
        """
        Grab all Palette defined at either EPOCH_START, ACQUISITION_POINT or intermediate with
        varying IDs.

        Returns
        -------

        list
            Contains all Palettes found in the global Palette definition at ACQUISITION_POINT
            or intermediate with varying IDs.
        """
        global_palettes: dict[int, list[Palette]] = {}
        for member in members:
            for pds_segment in member.pds_segments:
                if pds_segment.palette_id not in global_palettes:
                    global_palettes[pds_segment.palette_id] = pds_segment.palettes
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

            if ds.pcs.is_aquisition_point():
                reset_positions.append(index)

        return reset_positions

    def __find_redefintion_positions(
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
                    or ds.pcs.is_aquisition_point()
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

    def __process_timeline_item(
        self,
        new_timeline: TimelineItem,
        timelines: dict[str, list[TimelineItem]],
        ds: DisplaySet,
        global_palettes: dict[int, list[Palette]],
    ) -> dict[str, list[TimelineItem]]:
        """
        TimelineItems extracted from PGS subtitles have no correlation to their respective
        counterparts coming before or after.

        Process each item and extract the WindowID they are displayed in. If a prior item
        already exists within the Timelines dict, check if they are the same item referenced
        by their ID.

        If its a new item, simply add it to the Timelines dict, else update prior items data
        with current items data where required.

        Returns
        -------

        dict
            Timelines dict once a new item has been processed.
        """
        if new_timeline.position in timelines:
            prev_timeline = timelines[new_timeline.position][-1]
            prev_timeline.end = new_timeline.start

            if new_timeline.comp_obj.object_id != prev_timeline.comp_obj.object_id:
                if not new_timeline.palette:
                    new_timeline.palette = prev_timeline.palette

                if not new_timeline.display_obj:
                    new_timeline.display_obj = prev_timeline.display_obj

                timelines[new_timeline.position].append(new_timeline)
        else:
            if not new_timeline.palette:
                new_timeline.palette = global_palettes[ds.pcs.palette_id]
            timelines[new_timeline.position] = [new_timeline]

        return timelines

    def __gen_timelines(
        self, members: list[DisplaySet], global_palettes: dict[int, list[Palette]]
    ) -> dict[str, list[TimelineItem]]:
        """
        Generate timelines. Timelines consist of TimelineItems and describe the changes
        in either the Top or Bottom window of a PGS file. Items will be grouped as one
        if they display the same image within the same position and will be treated as
        new items if a new image is being defined.

        Returns
        -------

        dict
            Dictionary containing TimelineItems displayed in either Top or Bottom window.
        """
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
                        timelines = self.__process_timeline_item(
                            new_timeline, timelines, ds, global_palettes
                        )

        return timelines

    def __fix_endpoints(
        self,
        fixables: dict[str, list[TimelineItem]],
        reset_statements: DisplaySet,
        end: DisplaySet,
    ) -> dict[str, list[TimelineItem]]:
        """
        Reprocess dictionary containing TimelineItems displayed in either Top or Bottom window.
        Since END & RESET segments do not define images within them, they will not be correlated
        to a specific TimelineItem.

        However they define the true end timestamp for the TimelineItem prior, so the items end
        needs to be extended to match the END / RESET segments display timestamp.

        Returns
        -------

        dict
            Dictionary containing TimelineItems displayed in either Top or Bottom window.
        """
        for _, items in fixables.items():
            fixable = items[-1]

            if (
                reset_statements.pcs.composition_objects
                and fixable.comp_obj.object_id
                != reset_statements.pcs.composition_objects[0].object_id
            ):
                fixable.end = reset_statements.pcs.presentation_timestamp
            else:
                fixable.end = end.pcs.presentation_timestamp

        return fixables

    def __combine(
        self,
        previous: dict[str, list[TimelineItem]],
        current: dict[str, list[TimelineItem]],
        pos: str,
    ):
        prev = previous[pos][-1]
        curr = current[pos][0]
        if (
            prev.end == curr.start
            and prev.comp_obj.object_id == curr.comp_obj.object_id
        ):
            if not curr.display_obj:
                curr.display_obj = prev.display_obj

    def __look_to_combine(
        self, timelines: list[dict[str, list[TimelineItem]]]
    ) -> list[dict[str, list[TimelineItem]]]:
        previous: dict[str, list[TimelineItem]] | None = None
        for timeline in timelines:
            if previous is None:
                previous = timeline
                continue

            self.__combine(previous=previous, current=timeline, pos="Bottom")
            self.__combine(previous=previous, current=timeline, pos="Top")
            previous = timeline

        return timelines

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
        items: list[PgsSubtitleItem] = []

        for timeline in timelines:
            for _, entries in timeline.items():
                for element in entries:
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

    __slots__ = ("tmp_location", "temp_folder", "_items", "subtitle_groups")

    def __init__(
        self,
        tmp_location: str,
        temp_folder="tmp",
    ):
        self.tmp_location = tmp_location
        self.temp_folder = temp_folder
        self._items: typing.Optional[list[PgsSubtitleItem]] = None

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

        # Debug helper code
        # test_groups = list(range(100, 112))
        # test_groups = list(range(146, 149))
        # sliced = [ds for group in groups for ds in group if ds.index in test_groups]
        # self.subtitle_groups = [SubtitleGroup(members=sliced)]

        self.subtitle_groups = [SubtitleGroup(members=group) for group in groups]

        return list(
            chain.from_iterable([
                group.pgs_subtitle_items for group in self.subtitle_groups
            ])
        )

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
