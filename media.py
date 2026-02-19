from __future__ import annotations
from pgs import DisplaySet, PgsImage, ObjectDefinitionSegment, Palette
from PIL import Image
import typing
import logging


logger = logging.getLogger(__name__)


class PgsSubtitleItem:
    __slots__ = ("index", "start", "end", "image", "x_offset", "y_offset", "text", "place", "display_set")

    def __init__(self,
                 index: int,
                 start: float,
                 end: float,
                 display_set: DisplaySet,
                 ods: ObjectDefinitionSegment,
                 palette: list[Palette]
                 ):
        self.index = index
        self.start = start
        self.end = end
        self.display_set = display_set
        self.image = self.__generate_image(ods, palette)
        self.x_offset = display_set.wds.x_offset
        self.y_offset = display_set.wds.y_offset
        self.text: typing.Optional[str] = ""
        self.place: typing.Optional[typing.Tuple[int, int, int, int]] = (0, 0, 0, 0)


    def __generate_image(self, ods: ObjectDefinitionSegment, palettes: list[Palette]) -> PgsImage:
        return PgsImage(ods.img_data, palettes)

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

    def __repr__(self):
        return f'<{self.__class__.__name__} [{self}]>'

    def __str__(self):
        return f'[{self.start} --> {self.end or ""}]'
