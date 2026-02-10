from torch.multiprocessing import Manager, Pool, Process, set_start_method
from workers import OCRGPUWorker, LangaugeGPUWorker, CPUWorker
from subtitle_track_manager import SubtitleTrackManager
from model_core import OCRModelCore, LanguageModelCore
from itertools import chain
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from colorama import Fore
from pathlib import Path
import argparse
import logging
import torch
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog="PGS subtitle conversion using OCR and language identification on MKV files.",
        description="run python based PGS subtitle recognition",
    )
    parser.add_argument("-p", "--path", type=str, default="", help="Directory path to .mkv files. Will recursively scan subdirectories.")
    parser.add_argument("-o", "--override", type=bool, default=False, help="Override existing .srt file. Default: False")
    args = parser.parse_args()

    
    # Setup tmp directory and other parsed arguments
    tmp_path = Path(f"{os.path.dirname(os.path.realpath(__file__))}/tmp")
    if tmp_path.exists() == False:
        tmp_path.mkdir()
    options = {
        "path_to_tmp": "tmp",
        "override_if_exists": args.override
    }

    root = Path("test-files")
    if args.path:
        root = Path(args.path)


    # Get mkv files to extract subtitles from
    if root.is_file():
        convertibles = [root.absolute()]
    else:
        convertibles = (path.absolute() for path in root.rglob("*") if not path.is_dir() and ".mkv" in path.name)
    pgs_managers = chain.from_iterable((SubtitleTrackManager(file_path=path, options=options).get_pgs_managers() for path in convertibles))

    # Setup basic options relating to pytorch and set environmental variables if needed
    options["fallback_status"] = False
    try:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_device == "cuda":

            # Check for working rocm and activate flash attention, otherwise its NVIDIA
            if torch.version.hip != None:
                os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

        if torch.xpu.is_available():
            options["intel_disable_flash"] = True
            torch_device = "xpu"

        options["torch_device"] = torch_device
    except:
        options["fallback_status"] = True

    # Setup ocr prompt and message template
    task = "ocr"
    prompts = {
        "ocr": "OCR:",
    }
    message_template = [
                {"role": "user",         
                 "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": prompts[task]},
                    ]
                }
            ]
    
    try:
         set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    
    # Setup rich progressbar
    progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
    
    # Setup gpu processes and queues used for communication
    manager = Manager()
    queues  = {"ocr_queue": manager.Queue(), "pass_queue": manager.Queue(), "task_queue": manager.Queue(), "progress_queue": manager.Queue()}

    cpu_workers = 4
    for index in range(4, cpu_workers+4+1):
        queues[f"{index}"] = manager.Queue()

    gpu_processes = [
        Process(target=OCRGPUWorker(message_template=message_template, core=OCRModelCore, queues=queues, options=options).run, args=(1,)), # type: ignore
        Process(target=LangaugeGPUWorker(core=LanguageModelCore, queues=queues, options=options).run), # type: ignore
    ]
    [process.start() for process in gpu_processes]
    runnable = CPUWorker(queues,  options) # type: ignore

    try:
        with progress:
            with Pool(processes=cpu_workers) as pool:
                tasks = {}
                task_queue      = queues["task_queue"]
                progress_queue  = queues["progress_queue"]

                for _ in pool.imap_unordered(runnable.run, pgs_managers):
                    end = False
                    while end == False:
                        if task_queue.empty() == False:
                            description, total = task_queue.get_nowait()
                            task_id = progress.add_task(description=description, total=total, visible=True)

                            task = [task for task in progress.tasks if task.id == task_id][0]
                            tasks[description] = (task_id, task)


                        if progress_queue.empty() == False:
                            description = progress_queue.get_nowait()
                            if description in tasks:
                                task_id = tasks[description][0]
                                progress.update(task_id=task_id, advance=1)

                                task = tasks[description][1]

                                # Additionally have to check for if "task.remaining <= 1.0" as sometimes can get stuff with one missing
                                # Since file is still being saved and this only for fancy progressbar, should be ok
                                if task.finished:
                                    progress.update(task_id=task.id, visible=False)


                        # There should at least be a couple of tasks present, before we consider our progress finished. 
                        # Otherwise if the tool is tool slow, it will immidiately end the update loop
                        if progress.finished and not not tasks:
                            end = True

                queues["ocr_queue"].put((None, -1))

            [process.join() for process in gpu_processes]
            [process.close() for process in gpu_processes]
            
    except:
        [process.terminate() for process in gpu_processes]
                      
    
if __name__=="__main__":
    main()
