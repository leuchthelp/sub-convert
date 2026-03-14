from media import PgsSubtitleItem, Palette
from pgs import PgsReader, DisplaySet
from dataclasses import dataclass
from pysrt import SubRipTime
from copy import deepcopy
from colorama import Fore
import logging
import typing
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Timeline:

    def __init__(self,
                 window_id: int,
                 start: SubRipTime,
                 ds: DisplaySet,
                 index: int = 0,
                 ):
        
        self.window_id = window_id
        self.start = start
        self.end   = start # will be overwridden by the following Timeline item
        self.position =  "Bottom"

        self.ds    = ds
        self.comp_obj = [comp_obj for comp_obj in ds.pcs.composition_objects if comp_obj.window_id == window_id].pop()

        display_obj_cand = [display_obj for display_obj in ds.ods_segments if display_obj.id == self.comp_obj.object_id]
        self.display_obj = None if not display_obj_cand else display_obj_cand.pop()
        self.palette = None if not ds.pds_segments else ds.pds_segments.pop().palettes
        self.index = index
        self.pgs_subtitle_item: PgsSubtitleItem 


    def gen_pgs_subtitle_item(self) -> PgsSubtitleItem:
        if self.display_obj is None or self.palette is None:
            raise ValueError
        
        self.pgs_subtitle_item = PgsSubtitleItem(ods=self.display_obj, comp_obj=self.comp_obj, palette=self.palette)
        return self.pgs_subtitle_item


    def __repr__(self):
        return f'<{self.__class__.__name__} [{self}]>'


    def __str__(self):
        return f'[{self.start} --> {self.end or ""}]'


@dataclass
class SubtitleGroup:

    __slots__ = ("pgs_subtitle_items", "timelines")

    def __init__(self,
                 members: list[DisplaySet],
                 ):
        overlap = self.__find_overlap(members=members)

        end = members[-1]
        global_palettes: dict[int, list[Palette]] = {}
        timelines: list[dict[int, list[Timeline]]] = []

        if overlap:
            global_palettes = self.__find_global_palettes(members=members)
            redef_positions = self.__find_redefintion_positions(members=members)
            reset_positions = self.__find_reset_positions(members=members)
            overlapping     = self.__find_overlapping(reset_positions=reset_positions, redef_positions=redef_positions, members=members)

            tmp = [self.__gen_timelines(members=segment, global_palettes=global_palettes) for _, segment in overlapping.items()]
            reset_statements = [members[index] for index in reset_positions]
            redef_statements = [members[index] for index in redef_positions]

            for index, fixables in enumerate(tmp):
                actual_end = redef_statements[index + 1] if index == 0 else end
                timelines.append(self.__fix_endpoints(fixables=fixables, reset_statements=reset_statements[index], end=actual_end))
        else:
            tmp = self.__gen_timelines(members=members)
            timelines.append(self.__fix_endpoints(fixables=tmp, reset_statements=end, end=end))
        
        self.pgs_subtitle_items = self.__gen_pgs_subtitle_items(timelines=timelines)    


    def __find_overlap(self, members: list[DisplaySet]) -> bool:
        for ds in members:
            if ds.pcs.number_composition_objects > 1:
                return True
        return False


    def __find_global_palettes(self, members: list[DisplaySet]) -> dict[int, list[Palette]]:
        """
        Grab all Palette defined at either EPOCH_START, ACQUISITION_POINT or intermediate with varying IDs.
        
        Returns
        -------

        list
            Contains all Palettes found in the global Palette definition at ACQUISITION_POINT or intermediate with varying IDs
        """
        global_palettes: dict[int, list[Palette]] = {}
        for member in members:
            for pds_segment in member.pds_segments:
                if pds_segment.palette_id not in global_palettes:
                    global_palettes[pds_segment.palette_id] = pds_segment.palettes
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


    def __gen_timelines(self, members: list[DisplaySet], global_palettes: dict[int, list[Palette]] = {}) -> dict[int, list[Timeline]]:
        timelines: dict[int, list[Timeline]] = {}

        for ds in members:
            for comp_obj in ds.pcs.composition_objects:
                for window in ds.wds.windows:
                    if window.window_id == comp_obj.window_id:
                        if window.window_id in timelines:
                            prev_timeline = timelines[comp_obj.window_id][-1]
                            new_timeline = Timeline(window_id=comp_obj.window_id, start=ds.pcs.presentation_timestamp, index=prev_timeline.index + 1, ds=ds)

                            prev_timeline.end = new_timeline.start

                            if new_timeline.comp_obj.object_id != prev_timeline.comp_obj.object_id:
                                if not new_timeline.palette:
                                    new_timeline.palette = prev_timeline.palette

                                if not new_timeline.display_obj:
                                    new_timeline.display_obj = prev_timeline.display_obj

                                timelines[comp_obj.window_id].append(new_timeline) 
                        else:
                            new_timeline = Timeline(window_id=window.window_id, start=ds.pcs.presentation_timestamp, ds=ds) 

                            if not new_timeline.palette:
                                new_timeline.palette = global_palettes[ds.pcs.palette_id]
                            
                            timelines[window.window_id] = [new_timeline] 

        
        return timelines


    def __fix_endpoints(self, fixables: dict[int, list[Timeline]], reset_statements: DisplaySet, end: DisplaySet) -> dict[int, list[Timeline]]:
        for _, items in fixables.items():
            fixable = items[-1]
            
            if reset_statements.pcs.composition_objects and fixable.comp_obj.object_id != reset_statements.pcs.composition_objects[0].object_id:
                    fixable.end = reset_statements.pcs.presentation_timestamp
            else:
                fixable.end = end.pcs.presentation_timestamp

        return fixables


    def __gen_pgs_subtitle_items(self, timelines: list[dict[int, list[Timeline]]]) -> list[PgsSubtitleItem]:
        items: list[PgsSubtitleItem] = []

        for timeline in timelines:
            for _, entries in timeline.items():
                for element in entries:
                    items.append(element.gen_pgs_subtitle_item())
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

        
        test_groups = list(range(100, 112))
        sliced      = [ds for group in groups for ds in group if ds.index in test_groups]
        subtitle_groups = SubtitleGroup(members=sliced)

        return subtitle_groups.pgs_subtitle_items


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