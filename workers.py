from torch.multiprocessing import current_process, Queue
from model_core import OCRModelCore, LanguageModelCore
from pgs_manager import PgsManager, PgsSubtitleItem
from dataclasses import dataclass
from copy import deepcopy
from colorama import Fore
from pathlib import Path
import pytesseract as tess
import logging
import typing


logger = logging.getLogger(__name__)


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
        memory = {}
        while True:
            try:
                if not last_run_on_track:
                    image, return_queue = self.process_queue.get()

                    tmp_template = deepcopy(message_template)
                    tmp_template[0]["content"][0]["image"] = image
                    batch.append(tmp_template)
                    memory[str(len(batch) - 1)] = return_queue

                if len(batch) == batch_size:
                    last_run_on_track = False

                    if batch and memory:
                        texts = self.core.analyse(batch=batch)
                        logger.debug(f"{len(batch)}, {memory}, {texts}")
                        [
                            self.pass_queue.put_nowait(
                                (texts[int(index)], return_queue)
                            )
                            for index, return_queue in memory.items()
                        ]
                        batch.clear()
                        memory.clear()

            except:
                logger.debug(Fore.MAGENTA + "Last track" + Fore.RESET)
                batch_size = len(batch)
                last_run_on_track = True

    def __del__(self):
        del self.core


@dataclass
class LangaugeGPUWorker:
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
            original_text, return_queue = self.pass_queue.get()
            logger.debug(f"{original_text}, {return_queue}")

            text = str(original_text).lower()
            combined = self.core.get_topk(text=text)
            self.queues[return_queue].put_nowait((original_text, combined))

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
        "fallback",
        "override_if_exists",
    )

    def __init__(
        self,
        queues: dict[str, Queue],
        options={},
    ):
        self.gpu_ocr_queue = queues["ocr_queue"]
        self.queues = queues
        # Literally just for the progressbars to function as expected
        self.task_queue = queues["task_queue"]
        self.progress_queue = queues["progress_queue"]
        self.fallback = options["fallback_status"]
        self.override_if_exists = options["override_if_exists"]

    def run(self, pgs_manager: PgsManager) -> bool:
        pgs_data = pgs_manager.get_pgs_images()
        if not pgs_data:
            return False

        logger.debug(
            Fore.MAGENTA
            + f"Got to do {len(pgs_data)} items for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id} on {current_process()}"
            + Fore.RESET
        )
        self.task_queue.put_nowait(
            (
                f"[cyan]{pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}",
                len(pgs_data),
            )
        )

        queue_index = current_process().name.split("-")[1]
        return_queue = self.queues[f"{queue_index}"]

        logger.debug(
            Fore.LIGHTYELLOW_EX
            + f"{queue_index} got queue: {return_queue}"
            + Fore.RESET
        )
        finished: list[tuple[PgsSubtitleItem, str]] = []
        for image, item in pgs_data:
            test_width, test_height = image.size
            if test_width == 0 or test_height == 0:
                continue

            image = image.convert("RGB")
            text = ""
            if not self.fallback:
                self.gpu_ocr_queue.put_nowait((image, queue_index))
            else:
                text = tess.image_to_string(image=image)

            finished.append((item, text))

        for item, text in finished:
            self.progress_queue.put_nowait(
                (
                    f"[cyan]{pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}"
                )
            )

            combined: list[tuple[str, typing.Any]] = []
            if not self.fallback:
                text, combined = return_queue.get()

            if not text:
                continue

            item.text = text
            item.lang_estimate = combined

        logger.debug(
            Fore.GREEN
            + f"Finished extracting and classifying for {pgs_manager.hash[0:6]}-{Path(pgs_manager.mkv_track.file_path).name}-{pgs_manager.mkv_track.track_id}!"
            + Fore.RESET
        )
        pgs_manager.save_file()
        return True
