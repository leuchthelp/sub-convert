from torch.multiprocessing import Process, Manager, Pool, set_start_method
from workers import OCRGPUWorker, LangaugeGPUWorker, CPUWorker
from subtitle_track_manager import SubtitleTrackManager
from model_core import OCRModelCore, LanguageModelCore
from pstats import SortKey, Stats
from cProfile import Profile
from itertools import chain
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from pathlib import Path
import argparse
import logging
import torch
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_if_adjacent_exists(path: Path) -> bool:
    for file in Path(path.parent).glob("*"):
        tmp_name = str(path.name).replace(".mkv", "")
        logger.debug(f"{tmp_name} is in {file.name}: {tmp_name in file.name and path.name != file.name}?")
        if tmp_name in file.name and path.name != file.name:
            return True
    return False


def get_candidates(root: Path, options: dict):
    if root.is_file():
        yield root.absolute()

    for file in root.rglob("*.mkv"):
        if file.is_file():
            if options["skip_if_existing"]:
                if check_if_adjacent_exists(path=file) == False:
                    yield file.absolute()
            else:
                yield file.absolute()


def main():
    with Profile() as profile: 
        parser = argparse.ArgumentParser(
            prog="PGS subtitle conversion using OCR and language identification on MKV files.",
            description="run python based PGS subtitle recognition",
        )
        parser.add_argument("-p", "--path",             type=str, default="test-files", help="Directory path to .mkv files. Will recursively scan subdirectories.")
        parser.add_argument("-o", "--override",         type=bool, default=False, help="Override existing .srt file. Default: False")
        parser.add_argument("-s", "--skip_if_exists",   type=bool, default=False, help="Skip extracting and converting tracks if adjacent .srt track for file exist. Default: False")
        
        parser.add_argument("-c", "--cpu_workers",   type=int, default=4, help="Number of CPU workers. Default: 4")
        parser.add_argument("-ow", "--ocr_workers",  type=int, default=1, help="Number of OCR model workers, either on GPU or CPU. Default: 1")
        parser.add_argument("-lw", "--lang_workers", type=int, default=1, help="Number of Language model workers, either on GPU or CPU. Default: 1")
        parser.add_argument("-b", "--batchsize",     type=int, default=1, help="Size of the batch send to the OCR model. USE WITH CAUTION ON AMD GPU! Default: 1")
        args = parser.parse_args()

        # Setup tmp directory and other parsed arguments
        tmp_path = Path(f"{os.path.dirname(os.path.realpath(__file__))}/tmp")
        if tmp_path.exists() == False:
            tmp_path.mkdir()
        options = {
            "path_to_tmp": "tmp",
            "override_if_exists": args.override,
            "skip_if_existing"  : args.skip_if_exists,
        }
    
        root = Path(args.path)

    
        # Get mkv files to extract subtitles from
        convertibles = get_candidates(root=root, options=options)
        pgs_managers = chain.from_iterable((SubtitleTrackManager(file_path=path).get_pgs_managers(options=options) for path in convertibles))
    
    
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
        ocr_task = "ocr"
        prompts = {
            "ocr": "OCR:",
        }
        message_template = [
                    {"role": "user",         
                     "content": [
                            {"type": "image", "image": None},
                            {"type": "text", "text": prompts[ocr_task]},
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
        progress_manager = Manager()
        gpu_manager = Manager()
        queues  = {"ocr_queue": gpu_manager.Queue(), "pass_queue": gpu_manager.Queue(), "task_queue": progress_manager.Queue(), "progress_queue": progress_manager.Queue()}
    
    
        cpu_workers = args.cpu_workers
        gpu_ocr_workers = args.ocr_workers
        gpu_lang_workers = args.lang_workers
        for index in range(3+gpu_ocr_workers+gpu_lang_workers, 2+cpu_workers+gpu_ocr_workers+gpu_lang_workers+1):
            queues[f"{index}"] = progress_manager.Queue()
    
    
        gpu_ocr_batchsize = 1
        gpu_ocr_processes: list[Process] = []
        gpu_core = OCRModelCore(options=options)
        for _ in range(0, gpu_ocr_workers):
            gpu_ocr_processes.append(Process(target=OCRGPUWorker(core=gpu_core, queues=queues).run, args=(message_template, gpu_ocr_batchsize,))) # type: ignore
        del gpu_core
    
    
        gpu_lang_batchsize = 1
        gpu_lang_processes: list[Process] = []
        lang_core = LanguageModelCore(options=options)
        for _ in range(0, gpu_lang_workers):
            gpu_lang_processes.append(Process(target=LangaugeGPUWorker(core=lang_core, queues=queues).run, args=(gpu_lang_batchsize,))) # type: ignore
        del lang_core
    
        try:
            [process.start() for process in gpu_ocr_processes]
            [process.start() for process in gpu_lang_processes]
        
            runnable = CPUWorker(queues,  options) # type: ignore
            with progress:
                with Pool(processes=cpu_workers) as pool:
                    tasks = {}
                    task_queue      = queues["task_queue"]
                    progress_queue  = queues["progress_queue"]
                    del queues
    
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
                                        progress.update(task_id=task.id, refresh=True, visible=False)
    
    
                            # There should at least be a couple of tasks present, before we consider our progress finished. 
                            # Otherwise if the tool is tool slow, it will immidiately end the update loop
                            if progress.finished and not not tasks:
                                end = True
                
        except KeyboardInterrupt:
            pass
        
        finally:
            [process.terminate() for process in gpu_ocr_processes]
            [process.join() for process in gpu_ocr_processes]
            [process.close() for process in gpu_ocr_processes]
            
            [process.terminate() for process in gpu_lang_processes]
            [process.join() for process in gpu_lang_processes]
            [process.close() for process in gpu_lang_processes]
    


    stats = Stats(profile)#.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)
    stats.sort_stats(SortKey.CALLS).print_stats(20)
    stats.sort_stats(SortKey.TIME).print_stats(20)
                      
    
if __name__=="__main__":
    main()
