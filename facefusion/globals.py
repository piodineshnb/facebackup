import os
from typing import List, Optional
import psutil

# General
source_paths: Optional[List[str]] = None
reference_paths: Optional[List[str]] = None
target_path: Optional[str] = None
output_path: Optional[str] = None

# Misc
skip_download: Optional[bool] = True
headless: Optional[bool] = None
log_level: Optional[str] = 'info'  # Assuming LogLevel is a string-based enum

# Execution
execution_providers: List[str] = ['CUDAExecutionProvider']
execution_thread_count: Optional[int] = int(os.getenv('EXECUTION_THREAD_COUNT', str(psutil.cpu_count(logical=False))))
execution_queue_count: Optional[int] = int(os.getenv('EXECUTION_QUEUE_COUNT', '1'))

# Memory
video_memory_strategy: Optional[str] = 'strict'  # Assuming VideoMemoryStrategy is a string-based enum
system_memory_limit: Optional[int] = 0

# Face analyser
face_analyser_order: Optional[str] = 'left-right'  # Assuming FaceAnalyserOrder is a string-based enum
face_analyser_age: Optional[str] = None  # Assuming FaceAnalyserAge is a string-based enum
face_analyser_gender: Optional[str] = None  # Assuming FaceAnalyserGender is a string-based enum
face_detector_model: Optional[str] = 'yoloface'  # Assuming FaceDetectorModel is a string-based enum
face_detector_size: Optional[str] = '640x640'
face_detector_score: Optional[float] = 0.5
face_landmarker_score: Optional[float] = 0.5
face_recognizer_model: Optional[str] = 'arcface_inswapper'  # Assuming FaceRecognizerModel is a string-based enum

# Face selector
face_selector_mode: Optional[str] = 'reference'  # Assuming FaceSelectorMode is a string-based enum
reference_face_position: Optional[int] = 0
reference_face_distance: Optional[float] = 0.6
reference_frame_number: Optional[int] = 0

# Face mask
face_mask_types: Optional[List[str]] = ['occlusion']  # Assuming FaceMaskType is a string-based enum
face_mask_blur: Optional[float] = 0.3
face_mask_padding: Optional[tuple] = (0, 0, 0, 0)  # Assuming Padding is a tuple
face_mask_regions: Optional[List[str]] = ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'eye-glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']

# Frame extraction
trim_frame_start: Optional[int] = None
trim_frame_end: Optional[int] = None
temp_frame_format: Optional[str] = os.getenv('TEMP_FRAME_FORMAT', 'bmp')  # Assuming TempFrameFormat is a string-based enum
keep_temp: Optional[bool] = None

# Output creation
output_image_quality: Optional[int] = 80
output_image_resolution: Optional[str] = None
output_video_encoder: Optional[str] = os.getenv('OUTPUT_VIDEO_ENCODER', 'hevc_nvenc')  # Assuming OutputVideoEncoder is a string-based enum
retry_output_video_encoder: Optional[str] = os.getenv('RETRY_OUTPUT_VIDEO_ENCODER', 'libx264')  # Assuming OutputVideoEncoder is a string-based enum
output_video_preset: Optional[str] = os.getenv('OUTPUT_VIDEO_PRESET', 'veryfast') # Assuming OutputVideoPreset is a string-based enum
output_video_quality: Optional[int] = int(os.getenv('OUTPUT_VIDEO_QUALITY', '100'))
retry_output_video_quality: Optional[int] = int(os.getenv('RETRY_OUTPUT_VIDEO_QUALITY', '80'))
output_video_resolution: Optional[str] = None
output_video_fps: Optional[float] = None
skip_audio: Optional[bool] = None

# Frame processors
frame_processors: List[str] = ['face_swapper','face_enhancer']

# UIs
ui_layouts: List[str] = ['default']

# Extra
is_long_video: Optional[bool] = bool(int(os.getenv('IS_LONG_VIDEO', 0)))
update_interval: Optional[int] = int(os.getenv('UPDATE_INTERVAL_SECONDS', 120))
min_percentage_increment: Optional[int] = int(os.getenv('UPDATE_INTERVAL_PERCENTAGE', 10))
is_serverless: Optional[bool] = False
user_id: Optional[str] = None
session_id: Optional[str] = None
project_id: Optional[str] = None
output_media_url: Optional[str] = None
job_input: Optional[dict] = {}
swap_status: Optional[str] = 'processing'
last_updated: Optional[str] = None
credits: Optional[int] = 0 

# Progress cache for job status updates
progress_cache = {} 