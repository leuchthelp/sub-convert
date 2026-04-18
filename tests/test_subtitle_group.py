from pysrt import SubRipTime

from src.subtitle.subtitle_group import SubtitleGroup
from src.subtitle.subtitle_group import TimelineItem
from src.subtitle.subtitle_group import Pgs
from src.pgs.pgs_segments import PgsReader


def test_pgs():
    pgs = Pgs(tmp_location="tests/files/for-pgs/test.sup")
    assert bool(pgs.items) is True
    assert len(pgs.items) == 277


def test_subtitle_group():
    with open("tests/files/for-pgs/test.sup", "+rb") as data:
        display_sets = list(PgsReader.decode(data.read()))

    subtitle_group = SubtitleGroup(display_sets)

    assert subtitle_group.overlap is True
    assert len(subtitle_group.pgs_subtitle_items) == 24
    assert len(subtitle_group.timelines) == 4
    assert len(subtitle_group.timelines[0]["Top"]) == 1
    assert len(subtitle_group.timelines[0]["Bottom"]) == 1
    assert len(subtitle_group.timelines[1]["Top"]) == 1
    assert len(subtitle_group.timelines[1]["Bottom"]) == 3
    assert len(subtitle_group.timelines[2]["Top"]) == 4
    assert len(subtitle_group.timelines[2]["Bottom"]) == 3
    assert len(subtitle_group.timelines[3]["Top"]) == 7
    assert len(subtitle_group.timelines[3]["Bottom"]) == 4


def test_timeline_item():
    with open("tests/files/for-pgs/test.sup", "+rb") as data:
        display_sets = list(PgsReader.decode(data.read()))

    item = TimelineItem(start=SubRipTime(), ds=display_sets[0], window_id=0)

    assert item.gen_pgs_subtitle_item().height == 51
    assert item.gen_pgs_subtitle_item().width == 435