from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
import importlib
import argparse
import inspect
import logging
import os
import re


from torch.multiprocessing import Process, Manager, Pool, set_start_method
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from rich.progress import TaskID, Task


from src.model.workers import OCRGPUWorker, LanguageGPUWorker, CPUWorker
from src.subtitle.subtitle_track_manager import SubtitleTrackManager
from src.model import ocr_model_core, language_model_core


logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_if_adjacent_exists(path: Path) -> bool:
    tmp_name = str(path.name).replace(".mkv", "")

    for file in Path(path.parent).glob("*.srt"):
        if tmp_name in file.name:
            return True
    return False


def check_aged(path: Path, offset: str) -> bool:
    tmp = re.split("(\\W)", offset)

    delta = "h"
    try:
        if len(tmp) < 2:
            int_offset = int(tmp[0])
        else:
            if tmp[0]:
                delta = tmp[0]
            int_offset = int(tmp[1] + tmp[2])
    except KeyError as e:
        raise e
    except ValueError as e:
        raise ValueError(
            (
                f"Incorrect usage of -S, --skip_aged argument. {tmp} is not an"
                + "integer value. At least positive integer value is necessary!"
            )
        ) from e

    cutoff = datetime.now()
    match delta:
        case w if w in ["s", "S", "second", "Second", "seconds", "Seconds"]:
            cutoff = datetime.now() - timedelta(seconds=abs(int_offset))
        case w if w in ["m", "minute", "Minute", "minutes", "Minutes"]:
            cutoff = datetime.now() - timedelta(minutes=abs(int_offset))
        case w if w in ["h", "H", "hour", "Hour", "hours", "Hours"]:
            cutoff = datetime.now() - timedelta(hours=abs(int_offset))
        case w if w in ["d", "D", "day", "Day", "days", "Days"]:
            cutoff = datetime.now() - timedelta(days=abs(int_offset))
        case w if w in ["w", "W", "week", "Week", "weeks", "Weeks"]:
            cutoff = datetime.now() - timedelta(days=abs(int_offset))
        case w if w in ["M", "month", "Month", "months", "Months"]:
            cutoff = datetime.now() - timedelta(days=abs(int_offset * 30))
        case w if w in ["y", "Y", "year", "Year", "years", "Years"]:
            cutoff = datetime.now() - timedelta(days=abs(int_offset * 365))

    tmp_name = str(path.name).replace(".mkv", "")
    for file in Path(path.parent).glob("*.srt"):
        if tmp_name in file.name:
            file_age = datetime.fromtimestamp(file.stat().st_mtime)
            if (
                int_offset > 0
                and file_age < cutoff
                or int_offset < 0
                and file_age > cutoff
            ):
                return True
            return False
    return True


def get_candidates(root: Path, options: dict):
    if root.is_file():
        yield root.absolute()

    for file in root.rglob("*.mkv"):
        if file.is_file():
            if options["convert_aged"] and options["skip_if_existing"]:
                if not check_if_adjacent_exists(path=file) and check_aged(
                    path=file, offset=options["convert_aged"]
                ):
                    yield file.absolute()

            elif options["skip_if_existing"]:
                if not check_if_adjacent_exists(path=file):
                    yield file.absolute()

            elif options["convert_aged"]:
                if check_aged(path=file, offset=options["convert_aged"]):
                    yield file.absolute()
            else:
                yield file.absolute()


def get_classes(module) -> list[str]:
    return [
        cls.__name__
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    ]


def import_class(class_name: str, module_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def main():
    ocr_classes = get_classes(ocr_model_core)
    lang_classes = get_classes(language_model_core)

    parser = argparse.ArgumentParser(
        prog="PGS subtitle conversion using OCR and language identification on MKV files.",
        description="run python based PGS subtitle recognition",
    )
    parser.add_argument(
        "-om",
        "--ocr_model_core",
        choices=ocr_classes,
        default="OCRModelCore",
        help="List all options within the ocr_model_core.py "
        "which are the possible OCRModelCores to choose from.",
    )
    parser.add_argument(
        "-lm",
        "--language_model_core",
        choices=lang_classes,
        default="LanguageModelCore",
        help="List all options within the language_model_core.py "
        "which are the possible LanguageModelCores to choose from.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="files",
        help="Directory path to .mkv files. Will recursively scan subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing .srt file. Default: False",
    )
    parser.add_argument(
        "-s",
        "--skip_if_exists",
        action="store_true",
        help="Skip extracting and converting tracks if "
        "adjacent .srt track for file exist. Default: False",
    )
    parser.add_argument(
        "-a",
        "--convert_aged",
        type=str,
        default="",
        help=(
            "Extracting and converting tracks if older(+)/younger(-) than amount of offset from "
            + 'current date specified. Default: "", means nothing will be skipped. Given as str '
            + 'i.e. "H+8" = "process older than 8 hours". Will rerun the whole MKV file as soon as'
            + " it finds one SRT track +/- the threshold as SRT file-names generated by this tool "
            + "cannot be inferred back."
        ),
    )
    parser.add_argument(
        "-cw",
        "--cpu_workers",
        type=int,
        default=2,
        help="Number of CPU workers. Default: 2",
    )
    parser.add_argument(
        "-ow",
        "--ocr_workers",
        type=int,
        default=1,
        help="Number of OCR model workers, either on GPU or CPU. Default: 1",
    )
    parser.add_argument(
        "-lw",
        "--lang_workers",
        type=int,
        default=1,
        help="Number of Language model workers, either on GPU or CPU. Default: 1",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=1,
        help="Size of the batch send to the OCR model. USE WITH CAUTION ON AMD GPU! Default: 1",
    )
    parser.add_argument(
        "-d",
        "--dump-debug",
        action="store_true",
        help="Dumps debug info like a view of the timelines and PGS Displaysets under /debug/hash",
    )
    args = parser.parse_args()

    # Setup tmp directory and other parsed arguments
    tmp_path = Path(f"{os.path.dirname(os.path.realpath(__file__))}/tmp")
    if not tmp_path.exists():
        tmp_path.mkdir()
    options = {
        "path_to_tmp": "tmp",
        "overwrite_if_exists": args.overwrite,
        "skip_if_existing": args.skip_if_exists,
        "convert_aged": args.convert_aged,
        "dump_debug": args.dump_debug,
    }

    root = Path(args.path)

    # Get mkv files to extract subtitles from
    convertibles = get_candidates(root=root, options=options)
    pgs_managers = chain.from_iterable(
        (
            SubtitleTrackManager(file_path=path).get_pgs_managers(options=options)
            for path in convertibles
        )
    )

    #pgs_managers = [list(pgs_managers)[0]]

    # Setup ocr prompt and message template
    ocr_task = "ocr"
    prompts = {
        "ocr": "OCR:",
    }
    message_template = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompts[ocr_task]},
            ],
        }
    ]

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Setup rich progressbar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    # Setup gpu processes and queues used for communication
    progress_manager = Manager()
    gpu_manager = Manager()
    queues = {
        "ocr_queue": gpu_manager.Queue(),
        "pass_queue": gpu_manager.Queue(),
        "task_queue": progress_manager.Queue(),
        "progress_queue": progress_manager.Queue(),
    }

    cpu_workers = args.cpu_workers
    gpu_ocr_workers = args.ocr_workers
    gpu_lang_workers = args.lang_workers

    # Have to be a little careful with index counting as process numbering
    # cannot be set beforehand. As such need to ensure mapping of queue ids
    # to process ids
    for index in range(
        3 + gpu_ocr_workers + gpu_lang_workers,
        2 + cpu_workers + gpu_ocr_workers + gpu_lang_workers + 1,
    ):
        queues[f"{index}"] = progress_manager.Queue()

    gpu_ocr_batchsize = 1
    gpu_ocr_processes: list[Process] = []
    gpu_core_class = import_class(args.ocr_model_core, ocr_model_core.__name__)
    gpu_core = gpu_core_class(options=options)
    for idx in range(0, gpu_ocr_workers):
        gpu_ocr_processes.append(
            Process(
                target=OCRGPUWorker(core=gpu_core, queues=queues).run,
                name=f"OCRGPU{idx}",
                args=(
                    message_template,
                    gpu_ocr_batchsize,
                ),
            )
        )
    del gpu_core

    gpu_lang_batchsize = 1
    gpu_lang_processes: list[Process] = []
    language_core_class = import_class(
        args.language_model_core, language_model_core.__name__
    )
    lang_core = language_core_class(options=options)
    for idx in range(0, gpu_lang_workers):
        gpu_lang_processes.append(
            Process(
                target=LanguageGPUWorker(core=lang_core, queues=queues).run,
                name=f"LanguageGPU{idx}",
                args=(gpu_lang_batchsize,),
            )
        )
    del lang_core

    processes = gpu_ocr_processes + gpu_lang_processes

    try:
        for process in processes:
            process.start()

        runnable = CPUWorker(queues=queues)
        with progress:
            with Pool(processes=cpu_workers) as pool:
                tasks: dict[str, tuple[TaskID, Task]] = {}
                task_queue = queues["task_queue"]
                progress_queue = queues["progress_queue"]
                del queues

                for _ in pool.imap_unordered(runnable.run, pgs_managers):
                    end = False
                    while not end:
                        try:
                            description, total = task_queue.get_nowait()
                            task_id = progress.add_task(
                                description=description, total=total, visible=True
                            )

                            task = progress.tasks[int(task_id)]
                            tasks[description] = (task_id, task)
                        except Exception:
                            pass

                        try:
                            description = progress_queue.get_nowait()
                            if description in tasks:
                                task_id = tasks[description][0]
                                progress.update(
                                    task_id=task_id, advance=1, visible=True
                                )

                                task = tasks[description][1]
                                if task.finished:
                                    progress.update(
                                        task_id=task.id, refresh=True, visible=False
                                    )
                        except Exception:
                            pass

                        # There should at least be a couple of tasks present, before we
                        # consider our progress finished. Otherwise if the tool is tool
                        # slow, it will immidiately end the update loop
                        if progress.finished and tasks:
                            end = True

    except KeyboardInterrupt:
        pass

    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
            process.close()


if __name__ == "__main__":
    main()
