from media import PgsSubtitleItem, DisplaySet, Palette
from pgs import PgsReader
from dataclasses import dataclass
from colorama import Fore
import logging
import typing
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class SubtitleGroup:

    __slots__ = ("pgs_subtitle_items")

    #pgs_subtitle_items: list[PgsSubtitleItem]

    def __init__(self,
                 members: list[DisplaySet],
                 ):


        overlap = self.__find_overlap(members=members)

        global_palettes: dict[str, list[Palette]] = {}
        reset_positions: list[int] = []
        redef_positions: list[int] = []
        overlapping: dict[str, list[DisplaySet]] = {}
        if overlap:
            global_palettes = self.__find_global_palettes(members=members)
            reset_positions = self.__find_reset_positions(members=members)
            redef_positions = self.__find_redefintion_positions(members=members)
            overlapping     = self.__find_overlapping(reset_positions=reset_positions, redef_positions=redef_positions, members=members)
        

        #logger.info(Fore.YELLOW + f"members: {self.members}" + Fore.RESET)
        logger.info(Fore.MAGENTA + f"overlap: {overlap}, glo pal: {len(global_palettes)}, res pos: {reset_positions}, redef pods: {redef_positions}" + Fore.RESET)
        logger.info(Fore.CYAN + f"overlapping: {overlapping}" + Fore.RESET)
        
        self.pgs_subtitle_items = self.__generate_pgs_subtitles(members=members, overlap=overlap, global_palettes=global_palettes, overlapping=overlapping)

        


    def __find_overlap(self, members: list[DisplaySet]) -> bool:
        for ds in members:
            if ds.pcs.number_composition_objects > 1:
                return True
        return False
    

    def __find_global_palettes(self, members: list[DisplaySet]) -> dict[str, list[Palette]]:
        """
        Grab all Palette defined at either EPOCH_START, ACQUISITION_POINT or intermediate with varying IDs.
        
        Returns
        -------

        list
            Contains all Palettes found in the global Palette definition at ACQUISITION_POINT or intermediate with varying IDs
        """
        global_palettes: dict[str, list[Palette]] = {}
        for member in members:
            for pds_segment in member.pds_segments:
                if pds_segment.palette_id not in global_palettes:
                    global_palettes[str(pds_segment.palette_id)] = pds_segment.palettes
        return global_palettes


    def __find_reset_positions(self, members: list[DisplaySet]) -> list[int]:
        """
        In PGS Files END segments are usually sized to 11 bytes, contain no objects & are placed at the end of the group.
        However, if elements overlap an additional intermediate RESET segment is inserted which drops the number of objects
        and marks the position a Palette update can happen and either new Elements can overlap or the overlap ends.

        PGS subtitles can only show 2 objects on screen at once.

        This finds these positions. 

        They seem to always be marked with a size of 19 bytes and dropping the number of segments, which will always be 1.

        Returns
        -------

        list
            Contains the indices of the RESET segments
        """
        reset_positions = []
        for index, ds in enumerate(members):
            if ds.pcs.size == 19 and ds.pcs.number_composition_objects == 1 and not ds.ods_segments:
                reset_positions.append(index)
        return reset_positions


    def __find_redefintion_positions(self, members: list[DisplaySet]) -> list[int]:
        redef_positions = []
        for index, ds in enumerate(members):
            if ds.pcs.size == 19 and ds.pcs.number_composition_objects == 1 and ds.ods_segments and ds.pds_segments:
                redef_positions.append(index)
        return redef_positions


    def __find_overlapping(self, reset_positions: list[int], redef_positions: list[int], members: list[DisplaySet]) -> dict[str, list[DisplaySet]]:
        overlapping: dict[str, list[DisplaySet]] = {}

        for pos in range(len(reset_positions)):
            start = redef_positions[pos]
            stop  = reset_positions[pos]
            overlapping[str(start)] = members[start:stop + 1]
        return overlapping   
        

    def __generate_pgs_subtitles(self, overlap: bool, members: list[DisplaySet], global_palettes: dict[str, list[Palette]], overlapping: dict[str, list[DisplaySet]]) -> list[PgsSubtitleItem]:
        items: list[PgsSubtitleItem] = []
        for ds in members:
            for ods in ds.ods_segments:
                items.append(PgsSubtitleItem(0, 0.0, 0.0, ods=ods, palette=global_palettes[str(ds.pcs.palette_id)], display_set=ds))
        return items
    

class Pgs:
    __slots__ = ("data_reader", "temp_folder", "_items", "display_sets")

    def __init__(self,
                 data_reader: typing.Callable[[], bytes],
                 temp_folder="tmp",
                 ):
        self.data_reader = data_reader
        self.temp_folder = temp_folder
        self._items: typing.Optional[typing.List[PgsSubtitleItem]] = None

    @property
    def items(self) -> list[PgsSubtitleItem] | None:
        if self._items is None:
            data = self.data_reader()
            self._items = self.__decode(data)
        return self._items


    def __decode(self, data: bytes) -> list[PgsSubtitleItem]:
        display_sets = list(PgsReader.decode(data))
        groups: typing.List[typing.List[DisplaySet]] = []

        self.display_sets = display_sets

        tmp = []
        for ds in display_sets:
            if ds.is_start() or (ds.is_normal() and len(ds.ods_segments) != 0):
                tmp.append(ds)
            elif len(ds.ods_segments) == 0 and len(ds.pds_segments) == 0 and ds.is_normal():
                tmp.append(ds)
                groups.append(tmp)
                tmp = []

        
        test_groups = list(range(100, 111+1))
        sliced      = [ds for group in groups for ds in group if ds.index in test_groups]
        subtitle_groups = SubtitleGroup(members=sliced)

        return []

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