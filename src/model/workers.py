from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import logging
import typing

from torch.multiprocessing import current_process, Queue
from colorama import Fore

from pgs.pgs_manager import PgsManager, PgsSubtitleItem
from model.language_model_core import LanguageModelCore
from model.ocr_model_core import OCRModelCore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class OCRGPUWorker:
    __slots__ = ("process_queue", "pass_queue", "message_template", "core")

    def __init__(
        self,
        core: OCRModelCore,
        queues: dict[str, Queue],
    ):
        self.process_queue = queues["ocr_queue"]
        self.pass_queue = queues["pass_queue"]
        self.core = core
        del queues

    def run(self, message_template: list, batch_size=16):
        last_run_on_track = False
        batch = []
        memory: dict[int, tuple[str, int]] = {}
        while True:
            try:
                if not last_run_on_track:
                    image, return_queue, idx = self.process_queue.get()

                    tmp_template = deepcopy(message_template)
                    tmp_template[0]["content"][0]["image"] = image
                    batch.append(tmp_template)
                    memory[len(batch) - 1] = (return_queue, idx)

                if len(batch) == batch_size:
                    last_run_on_track = False

                    if batch and memory:
                        texts = self.core.analyse(batch=batch)
                        for index, (return_queue, idx) in memory.items():
                            self.pass_queue.put((texts[index], return_queue, idx))

                        batch.clear()
                        memory.clear()
            except Exception:
                batch_size = len(batch)
                last_run_on_track = True

    def __del__(self):
        del self.core


@dataclass
class LanguageGPUWorker:
    __slots__ = ("pass_queue", "queues", "core")

    def __init__(
        self,
        core: LanguageModelCore,
        queues: dict[str, Queue],
    ):
        self.pass_queue = queues["pass_queue"]
        self.queues = queues
        self.core = core

    def run(self, batch_size=16):
        end = False
        while not end:
            original_text, return_queue, idx = self.pass_queue.get()

            text = str(original_text).lower()
            combined = self.core.get_topk(text=text)
            self.queues[return_queue].put((original_text, combined, idx))

        logger.debug(Fore.MAGENTA + "LanguageGPUWorker ended" + Fore.RESET)

    def __del__(self):
        del self.core


@dataclass
class CPUWorker:
    __slots__ = (
        "gpu_ocr_queue",
        "queues",
        "task_queue",
        "progress_queue",
    )

    def __init__(
        self,
        queues: dict[str, Queue],
    ):
        self.gpu_ocr_queue = queues["ocr_queue"]
        self.queues = queues
        # Literally just for the progressbars to function as expected
        self.task_queue = queues["task_queue"]
        self.progress_queue = queues["progress_queue"]

    def run(self, pgs_manager: PgsManager) -> bool:
        pgs_data = pgs_manager.get_pgs_images()
        if not pgs_data:
            return False

        self.task_queue.put(
            (
                (
                    f"[cyan]{pgs_manager.hash[0:6]}"
                    + f"-{Path(pgs_manager.mkv_track.file_path).name}"
                    + f"-{pgs_manager.mkv_track.track_id}"
                ),
                len(pgs_data),
            )
        )

        queue_index = current_process().name.split("-")[1]
        return_queue = self.queues[f"{queue_index}"]

        finished: dict[int, PgsSubtitleItem] = {}
        for index, (image, item) in enumerate(pgs_data):
            test_width, test_height = image.size
            if test_width == 0 or test_height == 0:
                continue

            self.gpu_ocr_queue.put((image, queue_index, index))
            finished[index] = item

        safety_check = []
        for index in finished:
            combined: list[tuple[str, typing.Any]] = []
            text, combined, index = return_queue.get()

            self.progress_queue.put(
                (
                    f"[cyan]{pgs_manager.hash[0:6]}"
                    + f"-{Path(pgs_manager.mkv_track.file_path).name}"
                    + f"-{pgs_manager.mkv_track.track_id}"
                )
            )

            if not text:
                continue

            item = finished[index]
            item.text = text
            item.lang_estimate = combined
            safety_check.append(index)

        if not safety_check:
            return False

        pgs_manager.save_file()
        return True
