from pathlib import Path

from pysrt import SubRipTime
import py7zr

from src.subtitle.subtitle_group import SubtitleGroup
from src.subtitle.subtitle_group import TimelineItem
from src.subtitle.subtitle_group import Pgs
from src.pgs.pgs_segments import PgsReader


tmp_location = Path("tests/files/for-pgs/test.sup")
if not tmp_location.exists():
    with py7zr.SevenZipFile("tests/files/for-pgs/test.7z", mode='r') as archive:
        archive.extractall(path="tests/files/for-pgs/")

def test_pgs():
    pgs = Pgs(tmp_location=str(tmp_location))
    assert bool(pgs.items) is True
    assert len(pgs.items) == 277


def test_subtitle_group():
    with open(str(tmp_location), "+rb") as data:
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
    with open(str(tmp_location), "+rb") as data:
        display_sets = list(PgsReader.decode(data.read()))

    item = TimelineItem(start=SubRipTime(), ds=display_sets[0], window_id=0)

    assert item.gen_pgs_subtitle_item().height == 51
    assert item.gen_pgs_subtitle_item().width == 435

def test_last():
    tmp_location.unlink()
    assert tmp_location.exists() is False