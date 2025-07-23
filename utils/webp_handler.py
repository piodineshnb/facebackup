from PIL import Image, ImageSequence
from utils.custom_exception import CustomException
import moviepy.editor as mp
import tempfile

def webp_to_mp4(webp_path, output_path):
    try:
        with Image.open(webp_path) as img:
            frames = []
            durations = []

            # Iterate over each frame in the image sequence
            for frame in ImageSequence.Iterator(img):
                frames.append(frame.copy())
                duration = frame.info.get('duration', 0)
                if duration == 0:
                    duration = 100
                durations.append(duration)

        with tempfile.TemporaryDirectory() as temp_dir:
            clips = []
            for i, (frame, duration) in enumerate(zip(frames, durations)):
                frame_file = f"{temp_dir}/frame_{i}.png"
                frame.save(frame_file)
                clip = mp.ImageClip(frame_file).set_duration(duration / 1000)
                clips.append(clip)
                frame.close()  # Explicitly close each PIL image

            # Calculate FPS from the average duration of frames
            if sum(durations) != 0:
                fps = int(1000 / (sum(durations) / len(durations)))
            else:
                fps = 10
            fps = 10 if fps == 0 else fps
            video = mp.concatenate_videoclips(clips, method="compose")
            video.write_videofile(output_path, fps=fps)

            # Explicitly close and remove clips to free up memory
            for clip in clips:
                clip.close()
            clips.clear()

    except Exception as e:
        print(f"Error converting WEBP to MP4: {e}")
        raise CustomException(code="webp_to_mp4_failed", message="Error converting WEBP to MP4")
    finally:
        # Ensure all frames are cleared from memory
        del frames
        del durations

def mp4_to_webp(video_path, output_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mp.VideoFileClip(video_path) as clip:
                fps = clip.fps  # Assumes the fps has been correctly set during the MP4 creation

                frames = []
                for frame in clip.iter_frames(fps=fps, dtype='uint8'):
                    img = Image.fromarray(frame)
                    frames.append(img)

                frame_duration = 1000 / fps  # Duration in milliseconds
                frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)

                # Clear the frames list to free memory
                frames.clear()

    except Exception as e:
        print(f"Error converting MP4 to WEBP: {e}")
        raise CustomException(code="mp4_to_webp_failed",message="Error converting MP4 to WEBP")
