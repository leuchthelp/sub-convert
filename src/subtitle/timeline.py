from dataclasses import dataclass
import typing

from pysrt import SubRipTime

from pgs.pgs_subtitle_item import PgsSubtitleItem, Palette
from pgs.pgs_segments import (
    DisplaySet,
    WindowDefinitionSegment,
    PresentationCompositionSegment,
)


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
        comp_obj: PresentationCompositionSegment.CompositionObject | None = None,
        window: WindowDefinitionSegment.Window | None = None,
        ds: DisplaySet | None = None,
        end: SubRipTime = SubRipTime(),
    ):
        self.start = start
        self.end = end  # will be overwritten by the following TimelineItem item

        if ds is not None and window is not None and comp_obj is not None:
            self.comp_obj = comp_obj

            # Get DisplayObject (i.e an image), can also be empty meaning
            # it will reuse the image from the prior TimelineItem.
            display_obj_cand = [
                display_obj
                for display_obj in ds.ods_segments
                if display_obj.id == self.comp_obj.object_id
            ]
            self.display_obj = display_obj_cand

            # Full screen coordinates for PGS start in the top left;
            # smaller offset = higher up | larger offset = lower down
            position = "Bottom"
            border = ds.pcs.height / 2
            if window.height + self.comp_obj.y_offset < border:
                position = "Top"

            self.position = position

            self.palette = (
                None if not ds.pds_segments else ds.pds_segments.pop().palettes
            )

        self.pgs_subtitle_item: PgsSubtitleItem | None
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


def __process_timeline_item(
    new_timeline: TimelineItem,
    timelines: dict[str, list[TimelineItem]],
    ds: DisplaySet,
    global_palettes: dict[int, list[list[Palette]]],
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
            new_timeline.palette = global_palettes[ds.pcs.palette_id][0]
        timelines[new_timeline.position] = [new_timeline]

    return timelines


def gen_timelines(
    members: list[DisplaySet], global_palettes: dict[int, list[list[Palette]]]
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
                        window=window,
                        comp_obj=comp_obj,
                        start=ds.pcs.presentation_timestamp,
                        ds=ds,
                    )
                    timelines = __process_timeline_item(
                        new_timeline, timelines, ds, global_palettes
                    )

    return timelines


def fix_endpoints(
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
        if not reset_statements.pcs.composition_objects:
            fixable.end = end.pcs.presentation_timestamp
            break

        for obj in reset_statements.pcs.composition_objects:
            if fixable.comp_obj.object_id != obj.object_id:
                fixable.end = reset_statements.pcs.presentation_timestamp
            else:
                fixable.end = end.pcs.presentation_timestamp

    return fixables


def __combine(
    previous: dict[str, list[TimelineItem]],
    current: dict[str, list[TimelineItem]],
    pos: str,
):
    prev = previous[pos][-1]
    curr = current[pos][0]
    if prev.end == curr.start and prev.comp_obj.object_id == curr.comp_obj.object_id:
        if not curr.display_obj:
            curr.display_obj = prev.display_obj


def look_to_combine(
    timelines: list[dict[str, list[TimelineItem]]],
) -> list[dict[str, list[TimelineItem]]]:
    previous: dict[str, list[TimelineItem]] | None = None
    for timeline in timelines:
        if previous is None:
            previous = timeline
            continue

        if "Bottom" in timeline:
            __combine(previous=previous, current=timeline, pos="Bottom")

        if "Top" in timeline:
            __combine(previous=previous, current=timeline, pos="Top")
        previous = timeline

    return timelines
