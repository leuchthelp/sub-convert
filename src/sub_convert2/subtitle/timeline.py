import typing
from dataclasses import dataclass
from itertools import chain, groupby

from pysrt import SubRipTime

from sub_convert2.pgs.pgs_segments import (
    DisplaySet,
    PresentationCompositionSegment,
    WindowDefinitionSegment,
)
from sub_convert2.pgs.pgs_subtitle_item import Palette, PgsSubtitleItem


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
        og_comp_obj: list[PresentationCompositionSegment.CompositionObject]
        | None = None,
        window: WindowDefinitionSegment.Window | None = None,
        ds: DisplaySet | None = None,
        palette_id: int = 0,
        end: SubRipTime = SubRipTime(),  # noqa: B008
    ):
        self.ds = ds
        self.palette_id = palette_id
        self.window = window
        self.start = start
        self.end = end  # will be overwritten by the following TimelineItem item
        self.per_window_overlap_hint = False

        if og_comp_obj is not None and len(og_comp_obj) > 1:
            window_ids: list[int] = []
            for obj in og_comp_obj:
                window_ids.append(obj.window_id)

            g = groupby(window_ids)
            self.per_window_overlap_hint = next(g, True) and not next(g, False)

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
        if self.end < self.start:
            raise ValueError("End has not been set yet.")
        return self.end - self.start

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self}]>"

    def __str__(self):
        return f"[{self.start} --> {self.end or ''}]"


def __process_timeline_item(
    new_items: list[TimelineItem],
    timelines: dict[str, list[TimelineItem]],
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
    for item in new_items:
        if item.position in timelines:
            prev_timeline = timelines[item.position][-1]
            prev_timeline.end = item.start

            if item.comp_obj.object_id != prev_timeline.comp_obj.object_id:
                if not item.palette:
                    item.palette = prev_timeline.palette

                if not item.display_obj:
                    item.display_obj = prev_timeline.display_obj

                timelines[item.position].append(item)
        else:
            if not item.palette:
                item.palette = global_palettes[item.palette_id][0]
            timelines[item.position] = [item]

    return timelines


def __find_true_position(
    hints: list[TimelineItem],
) -> list[TimelineItem]:

    borders: dict[int, int] = {}
    for item in hints:
        window_id = item.comp_obj.window_id
        y_offset = item.comp_obj.y_offset
        if window_id not in borders:
            borders[window_id] = y_offset
        else:
            borders[window_id] = min(borders[window_id], y_offset)

    for item in hints:
        window_id = item.comp_obj.window_id
        y_offset = item.comp_obj.y_offset

        if y_offset <= borders[window_id]:
            item.position = "Top"
        else:
            item.position = "Bottom"

    if len(hints) > 1 and not hints[0].display_obj:
        hints.remove(hints[0])

    return hints


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

    per_window_id: dict[int, list[TimelineItem]] = {}
    timelines: dict[str, list[TimelineItem]] = {}
    for ds in members:
        for comp_obj in ds.pcs.composition_objects:
            for window in ds.wds.windows:
                if window.window_id == comp_obj.window_id:
                    item = TimelineItem(
                        window=window,
                        comp_obj=comp_obj,
                        og_comp_obj=ds.pcs.composition_objects,
                        start=ds.pcs.presentation_timestamp,
                        palette_id=ds.pcs.palette_id,
                        ds=ds,
                    )

                    if window.window_id not in per_window_id:
                        per_window_id[window.window_id] = [item]
                    else:
                        per_window_id[window.window_id].append(item)

    for window, items in per_window_id.items():
        hints: list[TimelineItem] = []
        residue: list[TimelineItem] = []
        for item in items:
            if item.per_window_overlap_hint:
                hints.append(item)
            else:
                residue.append(item)

        tmp = __find_true_position(hints)
        tmp.extend(residue)
        per_window_id[window] = tmp

    per_window_pos: dict[str, list[TimelineItem]] = {}
    for item in chain.from_iterable(per_window_id.values()):
        if item.position not in per_window_pos:
            per_window_pos[item.position] = [item]
        else:
            per_window_pos[item.position].append(item)

    for key, items in per_window_pos.items():
        per_window_pos[key] = sorted(items, key=lambda item: item.start)

    for items in per_window_pos.values():
        timelines = __process_timeline_item(items, timelines, global_palettes)

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
    for items in fixables.values():
        fixable = items[-1]
        if not reset_statements.pcs.composition_objects:
            fixable.end = end.pcs.presentation_timestamp
            break

        for obj in reset_statements.pcs.composition_objects:
            if fixable.comp_obj.object_id != obj.object_id:
                fixable.end = reset_statements.pcs.presentation_timestamp
            else:
                fixable.end = end.pcs.presentation_timestamp

        for display in reset_statements.ods_segments:
            if display.id == fixable.comp_obj.object_id:
                fixable.end = reset_statements.pcs.presentation_timestamp
                break

    return fixables


def __combine(
    previous: dict[str, list[TimelineItem]],
    current: dict[str, list[TimelineItem]],
    pos: str,
):
    prev = previous[pos][-1]
    curr = current[pos][0]
    if (
        prev.end == curr.start
        and prev.comp_obj.object_id == curr.comp_obj.object_id
        and not curr.display_obj
    ):
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

        if "Top" in timeline and "Top" in previous:
            __combine(previous=previous, current=timeline, pos="Top")
        previous = timeline

    return timelines
