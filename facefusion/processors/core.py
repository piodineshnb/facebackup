import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List

from tqdm import tqdm

from facefusion import logger, state_manager, wording
from facefusion.exit_helper import hard_exit
from facefusion.types import ProcessFrames, QueuePayload
from facefusion.globals import progress_cache

PROCESSORS_METHODS =\
[
	'get_inference_pool',
	'clear_inference_pool',
	'register_args',
	'apply_args',
	'pre_check',
	'pre_process',
	'post_process',
	'get_reference_frame',
	'process_frame',
	'process_frames',
	'process_image',
	'process_video'
]


def load_processor_module(processor : str) -> Any:
	try:
		processor_module = importlib.import_module('facefusion.processors.modules.' + processor)
		for method_name in PROCESSORS_METHODS:
			if not hasattr(processor_module, method_name):
				raise NotImplementedError
	except ModuleNotFoundError as exception:
		logger.error(wording.get('processor_not_loaded').format(processor = processor), __name__)
		logger.debug(exception.msg, __name__)
		hard_exit(1)
	except NotImplementedError:
		logger.error(wording.get('processor_not_implemented').format(processor = processor), __name__)
		hard_exit(1)
	return processor_module


def get_processors_modules(processors : List[str]) -> List[ModuleType]:
	processor_modules = []

	for processor in processors:
		processor_module = load_processor_module(processor)
		processor_modules.append(processor_module)
	return processor_modules


import sys

# These will be set before each stage starts
unified_state = {
    "swapper_total": 0,
    "swapper_current": 0,
    "enhancer_total": 0,
    "enhancer_current": 0,
    "stage": "swapper"  # or "enhancer"
}

# Add a global variable for total_faces
_total_faces_global = None

def set_total_faces(total_faces):
    global _total_faces_global
    _total_faces_global = total_faces
    print(f"[DEBUG] set_total_faces called, total_faces set to: {_total_faces_global}")

def write_progress_tempfile(percent):
    try:
        with open("temp/unified_progress.txt", "w") as f:
            f.write(str(percent))
    except Exception as e:
        print(f"[ERROR] Could not write unified progress file: {e}")

def read_progress_tempfile():
    try:
        with open("temp/unified_progress.txt", "r") as f:
            value = int(f.read().strip())
            return value
    except Exception as e:
        print(f"[ERROR] Could not read unified progress file: {e}")
        return 0

def delete_progress_tempfile():
    import os
    try:
        os.remove("temp/unified_progress.txt")
    except Exception:
        pass

def unified_progress_line(job_id=None, current_face_index=None, total_faces=None, face_progress=None):
    #print(f"[DEBUG] unified_progress_line: current_face_index={current_face_index}, total_faces={total_faces}, face_progress={face_progress}")
    #print(f"[DEBUG] unified_progress_line: os.environ.get('NUM_FACES') = {os.environ.get('NUM_FACES')}")
    #print(f"[DEBUG] unified_progress_line: os.environ.get('CURRENT_FACE_INDEX') = {os.environ.get('CURRENT_FACE_INDEX')}")
    # Use global total_faces if not provided
    if total_faces is None:
        if _total_faces_global is not None:
            total_faces = _total_faces_global
        else:
            # Try to get from environment variable NUM_FACES
            num_faces_env = os.environ.get("NUM_FACES")
            if num_faces_env is not None:
                try:
                    total_faces = int(num_faces_env)
                except Exception:
                    total_faces = None
    # New logic: single face swap, progress is 0-50 for swapper, 50-100 for enhancer
    if face_progress is not None:
        if unified_state["stage"] == "swapper":
            percent = int(face_progress * 50)
        elif unified_state["stage"] == "enhancer":
            percent = int(50 + face_progress * 50)
        else:
            percent = 0
    else:
        percent = 0
    sys.stdout.write(f"\rFaceFusion Unified Progress: {percent}%")
    sys.stdout.flush()
    write_progress_tempfile(percent)
    if job_id is not None:
        print(f"[DEBUG] Writing progress {percent}% to temp/progress_{job_id}.txt")
        try:
            with open(f"temp/progress_{job_id}.txt", "w") as f:
                f.write(str(percent))
        except Exception as e:
            print(f"[ERROR] Could not write progress file: {e}")


def get_unified_progress_percent(job_id):
    try:
        with open(f"temp/progress_{job_id}.txt", "r") as f:
            value = int(f.read().strip())
            print(f"[DEBUG] Read progress {value}% from temp/progress_{job_id}.txt")
            return value
    except Exception as e:
        print(f"[ERROR] Could not read progress file: {e}")
        return 0


def multi_process_frames(source_paths : List[str], temp_frame_paths : List[str], process_frames : ProcessFrames, job_id=None) -> None:
	queue_payloads = create_queue_payloads(temp_frame_paths)
	# Set the global state for this stage
	if "swapper" in process_frames.__module__:
		unified_state["stage"] = "swapper"
		unified_state["swapper_total"] = len(queue_payloads)
		unified_state["swapper_current"] = 0
	elif "enhancer" in process_frames.__module__:
		unified_state["stage"] = "enhancer"
		unified_state["enhancer_total"] = len(queue_payloads)
		unified_state["enhancer_current"] = 0

	def update_progress(n=1, face_index=None):
		if unified_state["stage"] == "swapper":
			unified_state["swapper_current"] += n
			face_progress = unified_state["swapper_current"] / unified_state["swapper_total"] if unified_state["swapper_total"] else 0
		elif unified_state["stage"] == "enhancer":
			unified_state["enhancer_current"] += n
			face_progress = unified_state["enhancer_current"] / unified_state["enhancer_total"] if unified_state["enhancer_total"] else 0
		else:
			face_progress = 0
		current_face_index = int(os.environ.get("CURRENT_FACE_INDEX", 0))
		#print(f"[DEBUG] update_progress: current_face_index={current_face_index}, face_progress={face_progress}")
		unified_progress_line(
			job_id=job_id,
			current_face_index=current_face_index,
			face_progress=face_progress
		)
		progress.update(n)  # keep tqdm for devs

	with tqdm(total = len(queue_payloads), desc = wording.get('processing'), unit = 'frame', ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		progress.set_postfix(execution_providers = state_manager.get_item('execution_providers'))
		with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
			futures = []
			queue : Queue[QueuePayload] = create_queue(queue_payloads)
			queue_per_future = max(len(queue_payloads) // state_manager.get_item('execution_thread_count') * state_manager.get_item('execution_queue_count'), 1)

			while not queue.empty():
				future = executor.submit(process_frames, source_paths, pick_queue(queue, queue_per_future), update_progress)
				futures.append(future)

			for future_done in as_completed(futures):
				future_done.result()


def create_queue(queue_payloads : List[QueuePayload]) -> Queue[QueuePayload]:
	queue : Queue[QueuePayload] = Queue()
	for queue_payload in queue_payloads:
		queue.put(queue_payload)
	return queue


def pick_queue(queue : Queue[QueuePayload], queue_per_future : int) -> List[QueuePayload]:
	queues = []
	for _ in range(queue_per_future):
		if not queue.empty():
			queues.append(queue.get())
	return queues


def create_queue_payloads(temp_frame_paths : List[str]) -> List[QueuePayload]:
	queue_payloads = []
	temp_frame_paths = sorted(temp_frame_paths, key = os.path.basename)

	for frame_number, frame_path in enumerate(temp_frame_paths):
		frame_payload : QueuePayload =\
		{
			'frame_number': frame_number,
			'frame_path': frame_path
		}
		queue_payloads.append(frame_payload)
	return queue_payloads
