import itertools
import shutil
import signal
import sys
from time import time

import numpy

from facefusion import cli_helper, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, logger, process_manager, state_manager, voice_extractor, wording
from facefusion.args import apply_args, collect_job_args, reduce_job_args, reduce_step_args
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_image, analyse_video
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.exit_helper import graceful_exit, hard_exit
from facefusion.face_analyser import get_average_face, get_many_faces, get_one_face
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import append_reference_face, clear_reference_faces, get_reference_faces
from facefusion.ffmpeg import copy_image, extract_frames, finalize_image, merge_video, replace_audio, restore_audio
from facefusion.filesystem import filter_audio_paths, get_file_name, is_image, is_video, resolve_file_paths, resolve_file_pattern
from facefusion.jobs import job_helper, job_manager, job_runner
from facefusion.jobs.job_list import compose_job_list
from facefusion.memory import limit_system_memory
from facefusion.processors.core import get_processors_modules
from facefusion.program import create_program
from facefusion.program_helper import validate_args
from facefusion.statistics import conditional_log_statistics
from facefusion.temp_helper import clear_temp_directory, create_temp_directory, get_temp_file_path, move_temp_file, resolve_temp_frame_paths
from facefusion.types import Args, ErrorCode
from facefusion.vision import pack_resolution, read_image, read_static_images, read_video_frame, restrict_image_resolution, restrict_trim_frame, restrict_video_fps, restrict_video_resolution, unpack_resolution


# --- API business logic migrated from api.py ---
import os
import uuid
import threading
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from facefusion.globals import progress_cache
from utils.check_and_hold_user_credits import check_and_hold_user_credits
from utils.deduct_user_credits import deduct_user_credits
from utils.refund_user_credits import refund_user_credits
from utils.register_video_swap_document import register_video_swap_document
from utils.update_video_swap_document import update_video_swap_document
from utils.register_error import register_error
from utils.download_file_from_url import download_file_from_url
from utils.upload_file import upload_file
from utils.remove_file import remove_file
from utils.update_progress import ProgressUpdater
from utils.update_swap_status_local import update_swap_status_local
from utils.custom_exception import CustomException
from facefusion.processors.core import set_total_faces, read_progress_tempfile

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
job_lock = threading.Lock()

def process_swap_job(params):
    credits_required = None
    temp_files_to_clean = []
    user_id = params.get("userId")
    swap_doc_id = params.get("swapId") or str(uuid.uuid4())
    project_id = params.get("projectId") or "default"
    target_url = params.get("mediaUrl")

    try:
        register_video_swap_document(user_id, swap_doc_id, params["faces"], project_id)
        credits_required = check_and_hold_user_credits(user_id, swap_doc_id, 0, project_id)

        original_video_path = os.path.join(TEMP_DIR, f"target_{swap_doc_id}.mp4")
        download_file_from_url(target_url, original_video_path)

        source_paths = []
        reference_paths = []
        for face in params["faces"]:
            source_path = os.path.join(TEMP_DIR, f"source_{uuid.uuid4()}.jpg")
            reference_path = os.path.join(TEMP_DIR, f"ref_{uuid.uuid4()}.jpg")
            download_file_from_url(face["source"], source_path)
            download_file_from_url(face["target"], reference_path)
            source_paths.append(source_path)
            reference_paths.append(reference_path)

        progress_updater = ProgressUpdater(update_interval=5, min_percentage_increment=5, user_id=user_id, session_id=swap_doc_id)
        progress_updater.set_total(len(source_paths))

        source_str = ",".join(source_paths)
        ref_str = ",".join(reference_paths)
        output_path = os.path.join(TEMP_DIR, f"result_{swap_doc_id}.mp4")

        # --- Start timing swap ---
        swap_start_time = time.time()
        cmd = [
            r"D:\facefusion\venv\Scripts\python.exe", "facefusion.py", "headless-run",
            "--target", original_video_path,
            "--output-path", output_path,
            "--face-enhancer-model", "gfpgan_1.4",
            "--face-enhancer-blend", "100",
            "--face-enhancer-weight", "1.0",
            "--output-video-encoder", "libx264",
            "--output-video-quality", "95",
            "--execution-providers", "cuda",
            "--source", source_str,
            "--reference-face-path", ref_str,
            "--processors", "face_swapper", "face_enhancer",
            "--face-selector-mode", "reference"
        ]

        env = os.environ.copy()
        env["ENABLE_FACE_ENHANCE"] = "1"
        env["NUM_FACES"] = str(len(source_paths))

        import re
        swap_time = None
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        all_output = []
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            all_output.append(line)
        process.wait()
        swap_end_time = time.time()
        # After process ends, join all output and search for the time string
        output_text = ''.join(all_output)
        print('DEBUG: Subprocess output for swap time parsing:')
        print(output_text)

        matches = re.findall(r'Processing to video succeed in\s*([0-9]+\.?[0-9]*)\s*seconds', output_text)
        #print('DEBUG: All matches:', matches)
        if matches:
            swap_time = float(matches[-1])
        else:
            swap_time = swap_end_time - swap_start_time
        # --- End timing swap ---

        if process.returncode != 0:
            raise Exception(f"FaceFusion failed with code {process.returncode}")

        # --- Generate preview and thumbnail ---
        from utils.create_webm_from_video import create_webm_from_video
        import tempfile
        preview_thumbnail_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
        preview_webm_path = create_webm_from_video(output_path, preview_thumbnail_path)

        # --- Start timing upload ---
        upload_start_time = time.time()

        # Upload files in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_preview = executor.submit(upload_file, file_path=preview_webm_path, file_name=os.path.basename(preview_webm_path), file_extension=".webm", project_id=project_id)
            future_thumbnail = executor.submit(upload_file, file_path=preview_thumbnail_path, file_name=os.path.basename(preview_thumbnail_path), file_extension="thumbnail", project_id=project_id)
            future_download = executor.submit(upload_file, file_path=output_path, file_name=os.path.basename(output_path), file_extension=".mp4", project_id=project_id)
            preview_url = future_preview.result()
            thumbnail_url = future_thumbnail.result()
            result_url = future_download.result()

        end_time = time.time()
        total_time = end_time - swap_start_time
        #print('DEBUG: t', total_time)
        upload_time = end_time - upload_start_time
        # --- End timing upload ---

        update_video_swap_document(user_id, result_url, swap_doc_id, preview_url, swap_time, upload_time, thumbnail_url, project_id=project_id)
        deduct_user_credits(user_id, credits_required, project_id)
        update_swap_status_local('completed')

        temp_files_to_clean.extend([original_video_path, *source_paths, *reference_paths, output_path, preview_webm_path, preview_thumbnail_path])
        return {"output": result_url, "credits_required": credits_required}, 200

    except CustomException as ce:
        register_error(user_id, swap_doc_id, {"error": ce.message}, project_id)
        if credits_required:
            refund_user_credits(user_id, credits_required, project_id)
        update_swap_status_local('failed')
        return {"error": ce.message}, 400

    except Exception as e:
        register_error(user_id, swap_doc_id, {"error": str(e)}, project_id)
        if credits_required:
            refund_user_credits(user_id, credits_required, project_id)
        update_swap_status_local('failed')
        return {"error": str(e)}, 500

    finally:
        for f in temp_files_to_clean:
            try:
                remove_file(f)
            except:
                pass

def execute_job(job_id, params, jobs):
    with job_lock:
        swap_doc_id = params.get("swapId")
        project_id = params.get('projectId', "default")
        print(f"Executing job {job_id}, swapId: {swap_doc_id}")
        response, status_code = process_swap_job(params)
        jobs[job_id] = {
            "status": "COMPLETED" if status_code == 200 else "FAILED",
            "output": response,
            "swapId": swap_doc_id,
            "projectId": project_id
        }
# --- END API business logic ---


def cli() -> None:
	if pre_check():
		signal.signal(signal.SIGINT, lambda signal_number, frame: graceful_exit(0))
		program = create_program()

		if validate_args(program):
			args = vars(program.parse_args())
			apply_args(args, state_manager.init_item)

			if state_manager.get_item('command'):
				logger.init(state_manager.get_item('log_level'))
				route(args)
			else:
				program.print_help()
		else:
			hard_exit(2)
	else:
		hard_exit(2)


def route(args : Args) -> None:
	system_memory_limit = state_manager.get_item('system_memory_limit')

	if system_memory_limit and system_memory_limit > 0:
		limit_system_memory(system_memory_limit)

	if state_manager.get_item('command') == 'force-download':
		error_code = force_download()
		return hard_exit(error_code)

	if state_manager.get_item('command') in [ 'job-list', 'job-create', 'job-submit', 'job-submit-all', 'job-delete', 'job-delete-all', 'job-add-step', 'job-remix-step', 'job-insert-step', 'job-remove-step' ]:
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_code = route_job_manager(args)
		hard_exit(error_code)

	if state_manager.get_item('command') == 'run':
		import facefusion.uis.core as ui

		if not common_pre_check() or not processors_pre_check():
			return hard_exit(2)
		for ui_layout in ui.get_ui_layouts_modules(state_manager.get_item('ui_layouts')):
			if not ui_layout.pre_check():
				return hard_exit(2)
		ui.init()
		ui.launch()

	if state_manager.get_item('command') == 'headless-run':
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_core = process_headless(args)
		hard_exit(error_core)

	if state_manager.get_item('command') == 'batch-run':
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_core = process_batch(args)
		hard_exit(error_core)

	if state_manager.get_item('command') in [ 'job-run', 'job-run-all', 'job-retry', 'job-retry-all' ]:
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_code = route_job_runner()
		hard_exit(error_code)


def pre_check() -> bool:
	if sys.version_info < (3, 10):
		logger.error(wording.get('python_not_supported').format(version = '3.10'), __name__)
		return False

	if not shutil.which('curl'):
		logger.error(wording.get('curl_not_installed'), __name__)
		return False

	if not shutil.which('ffmpeg'):
		logger.error(wording.get('ffmpeg_not_installed'), __name__)
		return False
	return True


def common_pre_check() -> bool:
	common_modules =\
	[
		content_analyser,
		face_classifier,
		face_detector,
		face_landmarker,
		face_masker,
		face_recognizer,
		voice_extractor
	]

	return all(module.pre_check() for module in common_modules)


def processors_pre_check() -> bool:
	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		if not processor_module.pre_check():
			return False
	return True


def force_download() -> ErrorCode:
	common_modules =\
	[
		content_analyser,
		face_classifier,
		face_detector,
		face_landmarker,
		face_masker,
		face_recognizer,
		voice_extractor
	]
	available_processors = [ get_file_name(file_path) for file_path in resolve_file_paths('facefusion/processors/modules') ]
	processor_modules = get_processors_modules(available_processors)

	for module in common_modules + processor_modules:
		if hasattr(module, 'create_static_model_set'):
			for model in module.create_static_model_set(state_manager.get_item('download_scope')).values():
				model_hash_set = model.get('hashes')
				model_source_set = model.get('sources')

				if model_hash_set and model_source_set:
					if not conditional_download_hashes(model_hash_set) or not conditional_download_sources(model_source_set):
						return 1

	return 0


def route_job_manager(args : Args) -> ErrorCode:
	if state_manager.get_item('command') == 'job-list':
		job_headers, job_contents = compose_job_list(state_manager.get_item('job_status'))

		if job_contents:
			cli_helper.render_table(job_headers, job_contents)
			return 0
		return 1

	if state_manager.get_item('command') == 'job-create':
		if job_manager.create_job(state_manager.get_item('job_id')):
			logger.info(wording.get('job_created').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(wording.get('job_not_created').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-submit':
		if job_manager.submit_job(state_manager.get_item('job_id')):
			logger.info(wording.get('job_submitted').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(wording.get('job_not_submitted').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-submit-all':
		if job_manager.submit_jobs(state_manager.get_item('halt_on_error')):
			logger.info(wording.get('job_all_submitted'), __name__)
			return 0
		logger.error(wording.get('job_all_not_submitted'), __name__)
		return 1

	if state_manager.get_item('command') == 'job-delete':
		if job_manager.delete_job(state_manager.get_item('job_id')):
			logger.info(wording.get('job_deleted').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(wording.get('job_not_deleted').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-delete-all':
		if job_manager.delete_jobs(state_manager.get_item('halt_on_error')):
			logger.info(wording.get('job_all_deleted'), __name__)
			return 0
		logger.error(wording.get('job_all_not_deleted'), __name__)
		return 1

	if state_manager.get_item('command') == 'job-add-step':
		step_args = reduce_step_args(args)

		if job_manager.add_step(state_manager.get_item('job_id'), step_args):
			logger.info(wording.get('job_step_added').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(wording.get('job_step_not_added').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-remix-step':
		step_args = reduce_step_args(args)

		if job_manager.remix_step(state_manager.get_item('job_id'), state_manager.get_item('step_index'), step_args):
			logger.info(wording.get('job_remix_step_added').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
			return 0
		logger.error(wording.get('job_remix_step_not_added').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-insert-step':
		step_args = reduce_step_args(args)

		if job_manager.insert_step(state_manager.get_item('job_id'), state_manager.get_item('step_index'), step_args):
			logger.info(wording.get('job_step_inserted').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
			return 0
		logger.error(wording.get('job_step_not_inserted').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-remove-step':
		if job_manager.remove_step(state_manager.get_item('job_id'), state_manager.get_item('step_index')):
			logger.info(wording.get('job_step_removed').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
			return 0
		logger.error(wording.get('job_step_not_removed').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
		return 1
	return 1


def route_job_runner() -> ErrorCode:
	if state_manager.get_item('command') == 'job-run':
		logger.info(wording.get('running_job').format(job_id = state_manager.get_item('job_id')), __name__)
		if job_runner.run_job(state_manager.get_item('job_id'), process_step):
			logger.info(wording.get('processing_job_succeed').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.info(wording.get('processing_job_failed').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-run-all':
		logger.info(wording.get('running_jobs'), __name__)
		if job_runner.run_jobs(process_step, state_manager.get_item('halt_on_error')):
			logger.info(wording.get('processing_jobs_succeed'), __name__)
			return 0
		logger.info(wording.get('processing_jobs_failed'), __name__)
		return 1

	if state_manager.get_item('command') == 'job-retry':
		logger.info(wording.get('retrying_job').format(job_id = state_manager.get_item('job_id')), __name__)
		if job_runner.retry_job(state_manager.get_item('job_id'), process_step):
			logger.info(wording.get('processing_job_succeed').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.info(wording.get('processing_job_failed').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-retry-all':
		logger.info(wording.get('retrying_jobs'), __name__)
		if job_runner.retry_jobs(process_step, state_manager.get_item('halt_on_error')):
			logger.info(wording.get('processing_jobs_succeed'), __name__)
			return 0
		logger.info(wording.get('processing_jobs_failed'), __name__)
		return 1
	return 2


def process_headless(args : Args) -> ErrorCode:
	job_id = job_helper.suggest_job_id('headless')
	step_args = reduce_step_args(args)

	if job_manager.create_job(job_id) and job_manager.add_step(job_id, step_args) and job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step):
		return 0
	return 1


def process_batch(args : Args) -> ErrorCode:
	job_id = job_helper.suggest_job_id('batch')
	step_args = reduce_step_args(args)
	job_args = reduce_job_args(args)
	source_paths = resolve_file_pattern(job_args.get('source_pattern'))
	target_paths = resolve_file_pattern(job_args.get('target_pattern'))

	if job_manager.create_job(job_id):
		if source_paths and target_paths:
			for index, (source_path, target_path) in enumerate(itertools.product(source_paths, target_paths)):
				step_args['source_paths'] = [ source_path ]
				step_args['target_path'] = target_path
				step_args['output_path'] = job_args.get('output_pattern').format(index = index)
				if not job_manager.add_step(job_id, step_args):
					return 1
			if job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step):
				return 0

		if not source_paths and target_paths:
			for index, target_path in enumerate(target_paths):
				step_args['target_path'] = target_path
				step_args['output_path'] = job_args.get('output_pattern').format(index = index)
				if not job_manager.add_step(job_id, step_args):
					return 1
			if job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step):
				return 0
	return 1


def process_step(job_id : str, step_index : int, step_args : Args) -> bool:
	clear_reference_faces()
	step_total = job_manager.count_step_total(job_id)
	step_args.update(collect_job_args())
	apply_args(step_args, state_manager.set_item)

	logger.info(wording.get('processing_step').format(step_current = step_index + 1, step_total = step_total), __name__)
	if common_pre_check() and processors_pre_check():
		error_code = conditional_process()
		return error_code == 0
	return False


def conditional_process() -> ErrorCode:
	start_time = time.time()

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		if not processor_module.pre_process('output'):
			return 2

	conditional_append_reference_faces()

	if is_image(state_manager.get_item('target_path')):
		return process_image(start_time)
	if is_video(state_manager.get_item('target_path')):
		return process_video(start_time)

	return 0


def conditional_append_reference_faces() -> None:
	if 'reference' in state_manager.get_item('face_selector_mode') and not get_reference_faces():
		reference_face_paths = state_manager.get_item('reference_face_paths')
		reference_face_path = state_manager.get_item('reference_face_path')
		
		if reference_face_paths:
			# Use multiple custom reference face images
			for i, ref_path in enumerate(reference_face_paths):
				reference_frame = read_image(ref_path)
				reference_faces = sort_and_filter_faces(get_many_faces([reference_frame]))
				reference_face = get_one_face(reference_faces, state_manager.get_item('reference_face_position'))
				if reference_face:
					append_reference_face(f'origin_{i}', reference_face)
		elif reference_face_path:
			# Use the single custom reference face image (backward compatibility)
			reference_frame = read_image(reference_face_path)
			reference_faces = sort_and_filter_faces(get_many_faces([reference_frame]))
			reference_face = get_one_face(reference_faces, state_manager.get_item('reference_face_position'))
			append_reference_face('origin', reference_face)
		else:
			source_frames = read_static_images(state_manager.get_item('source_paths'))
			source_faces = get_many_faces(source_frames)
			source_face = get_average_face(source_faces)
			if is_video(state_manager.get_item('target_path')):
				reference_frame = read_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
			else:
				reference_frame = read_image(state_manager.get_item('target_path'))
			reference_faces = sort_and_filter_faces(get_many_faces([ reference_frame ]))
			reference_face = get_one_face(reference_faces, state_manager.get_item('reference_face_position'))
			append_reference_face('origin', reference_face)

			if source_face and reference_face:
				for processor_module in get_processors_modules(state_manager.get_item('processors')):
					abstract_reference_frame = processor_module.get_reference_frame(source_face, reference_face, reference_frame)
					if numpy.any(abstract_reference_frame):
						abstract_reference_faces = sort_and_filter_faces(get_many_faces([ abstract_reference_frame ]))
						abstract_reference_face = get_one_face(abstract_reference_faces, state_manager.get_item('reference_face_position'))
						append_reference_face(processor_module.__name__, abstract_reference_face)


def process_image(start_time : float) -> ErrorCode:
	if analyse_image(state_manager.get_item('target_path')):
		return 3

	logger.debug(wording.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))
	logger.debug(wording.get('creating_temp'), __name__)
	create_temp_directory(state_manager.get_item('target_path'))

	process_manager.start()
	temp_image_resolution = pack_resolution(restrict_image_resolution(state_manager.get_item('target_path'), unpack_resolution(state_manager.get_item('output_image_resolution'))))
	logger.info(wording.get('copying_image').format(resolution = temp_image_resolution), __name__)
	if copy_image(state_manager.get_item('target_path'), temp_image_resolution):
		logger.debug(wording.get('copying_image_succeed'), __name__)
	else:
		logger.error(wording.get('copying_image_failed'), __name__)
		process_manager.end()
		return 1

	temp_file_path = get_temp_file_path(state_manager.get_item('target_path'))
	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		logger.info(wording.get('processing'), processor_module.__name__)
		processor_module.process_image(state_manager.get_item('source_paths'), temp_file_path, temp_file_path)
		processor_module.post_process()
	if is_process_stopping():
		process_manager.end()
		return 4

	logger.info(wording.get('finalizing_image').format(resolution = state_manager.get_item('output_image_resolution')), __name__)
	if finalize_image(state_manager.get_item('target_path'), state_manager.get_item('output_path'), state_manager.get_item('output_image_resolution')):
		logger.debug(wording.get('finalizing_image_succeed'), __name__)
	else:
		logger.warn(wording.get('finalizing_image_skipped'), __name__)

	logger.debug(wording.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))

	if is_image(state_manager.get_item('output_path')):
		seconds = '{:.2f}'.format((time.time() - start_time) % 60)
		logger.info(wording.get('processing_image_succeed').format(seconds = seconds), __name__)
		conditional_log_statistics()
	else:
		logger.error(wording.get('processing_image_failed'), __name__)
		process_manager.end()
		return 1
	process_manager.end()
	return 0


def process_video(start_time : float) -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	if not analyse_video(state_manager.get_item('target_path'), trim_frame_start, trim_frame_end):
		return 3

	logger.debug(wording.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))
	logger.debug(wording.get('creating_temp'), __name__)
	create_temp_directory(state_manager.get_item('target_path'))

	process_manager.start()
	temp_video_resolution = pack_resolution(restrict_video_resolution(state_manager.get_item('target_path'), unpack_resolution(state_manager.get_item('output_video_resolution'))))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	logger.info(wording.get('extracting_frames').format(resolution = temp_video_resolution, fps = temp_video_fps), __name__)
	if extract_frames(state_manager.get_item('target_path'), temp_video_resolution, temp_video_fps, trim_frame_start, trim_frame_end):
		logger.debug(wording.get('extracting_frames_succeed'), __name__)
	else:
		if is_process_stopping():
			process_manager.end()
			return 4
		logger.error(wording.get('extracting_frames_failed'), __name__)
		process_manager.end()
		return 1

	temp_frame_paths = resolve_temp_frame_paths(state_manager.get_item('target_path'))
	if temp_frame_paths:
		for processor_module in get_processors_modules(state_manager.get_item('processors')):
			logger.info(wording.get('processing'), processor_module.__name__)
			processor_module.process_video(state_manager.get_item('source_paths'), temp_frame_paths)
			processor_module.post_process()
		if is_process_stopping():
			return 4
	else:
		logger.error(wording.get('temp_frames_not_found'), __name__)
		process_manager.end()
		return 1

	logger.info(wording.get('merging_video').format(resolution = state_manager.get_item('output_video_resolution'), fps = state_manager.get_item('output_video_fps')), __name__)
	if merge_video(state_manager.get_item('target_path'), temp_video_fps, state_manager.get_item('output_video_resolution'), state_manager.get_item('output_video_fps'), trim_frame_start, trim_frame_end):
		logger.debug(wording.get('merging_video_succeed'), __name__)
	else:
		if is_process_stopping():
			process_manager.end()
			return 4
		logger.error(wording.get('merging_video_failed'), __name__)
		process_manager.end()
		return 1

	if state_manager.get_item('output_audio_volume') == 0:
		logger.info(wording.get('skipping_audio'), __name__)
		move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	else:
		source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
		if source_audio_path:
			if replace_audio(state_manager.get_item('target_path'), source_audio_path, state_manager.get_item('output_path')):
				logger.debug(wording.get('replacing_audio_succeed'), __name__)
			else:
				if is_process_stopping():
					process_manager.end()
					return 4
				logger.warn(wording.get('replacing_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
		else:
			if restore_audio(state_manager.get_item('target_path'), state_manager.get_item('output_path'), trim_frame_start, trim_frame_end):
				logger.debug(wording.get('restoring_audio_succeed'), __name__)
			else:
				if is_process_stopping():
					process_manager.end()
					return 4
				logger.warn(wording.get('restoring_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))

	logger.debug(wording.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))

	if is_video(state_manager.get_item('output_path')):
		seconds = '{:.2f}'.format((time.time() - start_time))
		logger.info(wording.get('processing_video_succeed').format(seconds = seconds), __name__)
		conditional_log_statistics()
	else:
		logger.error(wording.get('processing_video_failed'), __name__)
		process_manager.end()
		return 1
	process_manager.end()
	return 0


def is_process_stopping() -> bool:
	if process_manager.is_stopping():
		process_manager.end()
		logger.info(wording.get('processing_stopped'), __name__)
	return process_manager.is_pending()
