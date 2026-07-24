import logging
import typing

from sub_convert2.pgs.pgs_segments import (
    ObjectDefinitionSegment,
    Palette,
    PgsImage,
    PresentationCompositionSegment,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PgsSubtitleItem:
    __slots__ = (
        "display_set",
        "image",
        "lang_estimate",
        "text",
        "x_offset",
        "y_offset",
    )

    def __init__(
        self,
        ods: list[ObjectDefinitionSegment],
        comp_obj: PresentationCompositionSegment.CompositionObject,
        palette: list[Palette],
    ):
        self.image = self.__generate_pgs_image(ods, palette)
        self.x_offset = comp_obj.x_offset
        self.y_offset = comp_obj.y_offset
        self.text: str = ""
        self.lang_estimate: list[tuple[str, typing.Any]] = []

    def __generate_pgs_image(
        self, ods: list[ObjectDefinitionSegment], palettes: list[Palette]
    ) -> PgsImage:
        tmp = b""
        for entry in ods:
            tmp += entry.img_data
        return PgsImage(data=tmp, palettes=palettes)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def h_center(self):
        shape = self.shape
        return shape[0] + (shape[2] - shape[0]) // 2

    @property
    def shape(self) -> tuple[int, ...]:
        height, width = self.height, self.width
        y_offset, x_offset = self.y_offset, self.x_offset

        return y_offset, x_offset, y_offset + height, x_offset + width
