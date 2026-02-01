from __future__ import annotations

import json
import logging
import os
import shutil
import typing
from pgs import DisplaySet, Palette, PgsImage, PgsReader
from utils import pairwise

logger = logging.getLogger(__name__)


class PgsSubtitleItem:

    def __init__(self,
                 index: int,
                 display_sets: typing.List[DisplaySet]):
        self.index = index
        self.start = min([ds.pcs.presentation_timestamp for ds in display_sets])
        self.end = max([ds.pcs.presentation_timestamp for ds in display_sets])
        self.image = PgsSubtitleItem.generate_image(display_sets)
        self.x_offset = min([ds.wds.x_offset for ds in display_sets if ds.wds.num_windows > 0])
        self.y_offset = min([ds.wds.y_offset for ds in display_sets if ds.wds.num_windows > 0])
        self.text: typing.Optional[str] = None
        self.place: typing.Optional[typing.Tuple[int, int, int, int]] = None

    @staticmethod
    def create_items(display_sets: typing.Iterable[DisplaySet]):
        current_sets: typing.List[DisplaySet] = []
        index = 0
        candidates: typing.List[PgsSubtitleItem] = []
        for ds in display_sets:
            if current_sets and ds.is_start():
                candidates.append(PgsSubtitleItem(index, current_sets))
                current_sets = []
                index += 1

            current_sets.append(ds)

        if current_sets:
            candidates.append(PgsSubtitleItem(index, current_sets))

        results = []
        for item, next_item in pairwise(candidates):
            if item.auto_fix(next_item=next_item):
                results.append(item)

        return results

    @staticmethod
    def generate_image(display_sets: typing.Iterable[DisplaySet]):
        for ds in display_sets:
            if not ds.pcs.is_start():
                continue

            palettes: typing.List[Palette] = []
            for pds in ds.pds_segments:
                palettes += pds.palettes
            img_data = b''
            for ods in ds.ods_segments:
                img_data += ods.img_data

            return PgsImage(img_data, palettes)

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def h_center(self):
        shape = self.shape
        return shape[0] + (shape[2] - shape[0]) // 2

    @property
    def shape(self):
        height, width = self.height, self.width
        y_offset, x_offset = self.y_offset, self.x_offset

        return y_offset, x_offset, y_offset + height, x_offset + width

    def auto_fix(self, next_item: typing.Optional[PgsSubtitleItem]):
        valid = True
        if self.image is None:
            logger.warning('Corrupted %r: No Image', self)
            valid = False
        if self.y_offset is None:
            logger.warning('Corrupted %r: No y_offset', self)
            valid = False
        if self.x_offset is None:
            logger.warning('Corrupted %r: No x_offset', self)
            valid = False
        if self.start is None:
            logger.warning('Corrupted %r: No Start timestamp', self)
            valid = False
        elif not self.end or self.end <= self.start:
            if next_item and next_item.start and self.start + 10000 >= next_item.start:
                self.end = max(self.start + 1, next_item.start - 1)
                logger.info('Fix applied for %r: Subtitle end timestamp was fixed', self)
            else:
                logger.warning('Corrupted %r: Subtitle with corrupted end timestamp', self)
                valid = False

        return valid

    def intersect(self, item: PgsSubtitleItem):
        shape = self.shape

        return shape[0] <= item.h_center <= shape[2]

    def __repr__(self):
        return f'<{self.__class__.__name__} [{self}]>'

    def __str__(self):
        return f'[{self.start} --> {self.end or ""}]'


class Pgs:

    def __init__(self,
                 data_reader: typing.Callable[[], bytes],
                 temp_folder="",
                 ):
        self.data_reader = data_reader
        self.temp_folder = temp_folder
        self._items: typing.Optional[typing.List[PgsSubtitleItem]] = None

    @property
    def items(self) -> list:
        if self._items is None:
            data = self.data_reader()
            self._items = self.decode(data)
        return self._items


    def decode(self, data: bytes) -> PgsSubtitleItem:
        display_sets = list(PgsReader.decode(data))
        self.display_sets = display_sets

        return PgsSubtitleItem.create_items(display_sets)

    def dump_display_sets(self, display_sets: typing.List[DisplaySet]):
        new_line = '\n'
        with open(os.path.join(self.temp_folder, 'display-sets.txt'), mode='w', encoding='utf8') as f:
            f.write(f'{new_line.join([str(ds) for ds in display_sets])}')
        with open(os.path.join(self.temp_folder, 'display-sets.json'), mode='w', encoding='utf8') as f:
            json.dump([ds.to_json() for ds in display_sets], f,
                      indent=2, ensure_ascii=False, default=lambda x: str(x))

    def __repr__(self):
        return f'<{self.__class__.__name__} [{self}]>'

    def __enter__(self):
        return self

    def __exit__(self):
        self._items = None
        keep = True
        if keep:
            logger.info('Keeping temporary files in %s', self.temp_folder)
        else:
            logger.debug('Removing temporary files in %s', self.temp_folder)
            shutil.rmtree(self.temp_folder)
