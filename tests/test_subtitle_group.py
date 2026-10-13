from pathlib import Path

import py7zr
from pysrt import SubRipTime

from sub_convert2.pgs.pgs_segments import PgsReader
from sub_convert2.subtitle.subtitle_group import Pgs, SubtitleGroup, TimelineItem

tmp_location_with_overlapping = Path("tests/files/for-pgs/test.sup")
if not tmp_location_with_overlapping.exists():
    with py7zr.SevenZipFile(
        "tests/files/for-pgs/test_with_overlapping.7z", mode="r"
    ) as archive:
        archive.extractall(path="tests/files/for-pgs/")


def test_pgs_with_overlapping():
    pgs = Pgs(tmp_location=str(tmp_location_with_overlapping))
    assert bool(pgs.items) is True
    assert len(pgs.items) == 277


def test_subtitle_group_with_overlapping():
    with open(str(tmp_location_with_overlapping), "+rb") as data:
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


def test_timeline_item_with_overlapping():
    with open(str(tmp_location_with_overlapping), "+rb") as data:
        display_sets = list(PgsReader.decode(data.read()))

    ds = display_sets[0]
    comp_obj = ds.pcs.composition_objects[-1]
    window = ds.wds.windows[-1]
    item = TimelineItem(start=SubRipTime(), comp_obj=comp_obj, window=window, ds=ds)

    assert item.gen_pgs_subtitle_item().height == 51
    assert item.gen_pgs_subtitle_item().width == 435


tmp_location_with_n_images_per_window_forced = Path(
    "tests/files/for-pgs/test_with_n_images_per_window_forced.sup"
)
if not tmp_location_with_n_images_per_window_forced.exists():
    with py7zr.SevenZipFile(
        "tests/files/for-pgs/test_with_n_images_per_window_forced.7z", mode="r"
    ) as archive:
        archive.extractall(path="tests/files/for-pgs/")


def test_pgs_with_n_images_per_window_forced():
    pgs = Pgs(tmp_location=str(tmp_location_with_n_images_per_window_forced))
    assert bool(pgs.items) is True
    assert len(pgs.items) == 80


def test_subtitle_group_with_n_images_per_window_forced():
    pgs = Pgs(str(tmp_location_with_n_images_per_window_forced))
    
    assert(len(pgs.items)) == 80
    subtitle_groups = pgs.subtitle_groups
    assert len(subtitle_groups) == 16

    assert subtitle_groups[0].overlap is False
    assert len(subtitle_groups[0].pgs_subtitle_items) == 1
    assert len(subtitle_groups[0].timelines) == 1
    assert len(subtitle_groups[0].timelines[0]["Bottom"]) == 1
    assert len(subtitle_groups[1].timelines) == 12
    assert len(subtitle_groups[1].timelines[0]["Bottom"]) == 1
    assert len(subtitle_groups[2].timelines[0]["Bottom"]) == 1
    assert len(subtitle_groups[3].timelines[0]["Bottom"]) == 1
    assert len(subtitle_groups[4].timelines[0]["Bottom"]) == 1
    assert len(subtitle_groups[5].timelines[0]["Bottom"]) == 1
    assert subtitle_groups[6].timelines[0]["Top"][0].start == SubRipTime(0, 12, 25, 870)
    assert len(subtitle_groups[6].timelines[0]["Top"]) == 1
    assert len(subtitle_groups[7].timelines[0]["Bottom"]) == 1


def test_timeline_with_n_images_per_window_forced():
    with open(str(tmp_location_with_n_images_per_window_forced), "+rb") as data:
        display_sets = list(PgsReader.decode(data.read()))

    ds = display_sets[0]
    comp_obj = ds.pcs.composition_objects[-1]
    window = ds.wds.windows[-1]
    item = TimelineItem(start=SubRipTime(), comp_obj=comp_obj, window=window, ds=ds)

    assert item.gen_pgs_subtitle_item().height == 237
    assert item.gen_pgs_subtitle_item().width == 717


def test_last():
    tmp_location_with_overlapping.unlink()
    assert tmp_location_with_overlapping.exists() is False

    tmp_location_with_n_images_per_window_forced.unlink()
    assert tmp_location_with_n_images_per_window_forced.exists() is False
