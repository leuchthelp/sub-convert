from pgs.pgs_segments import (
    PgsImage,
    ObjectDefinitionSegment,
    Palette,
    PresentationCompositionSegment,
)
import logging
import typing


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PgsSubtitleItem:
    __slots__ = (
        "image",
        "x_offset",
        "y_offset",
        "text",
        "display_set",
        "lang_estimate",
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
