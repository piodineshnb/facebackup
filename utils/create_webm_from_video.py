from moviepy.editor import VideoFileClip
import os
import uuid
import tempfile
from PIL import Image


def create_webm_from_video(video_path,thumbnail_path, start_time=0, duration=5, target_resolution=(480, 360), threads=4):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    actual_duration = clip.duration
    if start_time + duration > actual_duration:
        duration = actual_duration - start_time

    current_width, current_height = clip.size
    if current_width > target_resolution[0] or current_height > target_resolution[1]:
        resize_factor = min(target_resolution[0] / current_width, target_resolution[1] / current_height)
        clip = clip.resize(newsize=(int(current_width * resize_factor), int(current_height * resize_factor)))

    clip = clip.subclip(start_time, start_time + duration)

    target_fps = 20 if fps > 20 else fps
    clip = clip.set_fps(target_fps)

    webm_filename = f"{uuid.uuid4()}.webm"
    webm_path = os.path.join(tempfile.gettempdir(), webm_filename)
    clip.write_videofile(webm_path, codec='libvpx', fps=target_fps, threads=threads)
    # Extract the first frame and save it as a thumbnail
    first_frame = clip.get_frame(0)
    image = Image.fromarray(first_frame)
    image.save(thumbnail_path, format='JPEG', quality=90)

    clip.close()

    return webm_path
