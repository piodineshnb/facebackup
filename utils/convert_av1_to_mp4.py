import uuid
import subprocess
from utils.custom_exception import CustomException
import time


def convert_av1_to_mp4(input_path):
    """Convert AV1 video to MP4 format using ffmpeg with GPU acceleration"""
    output_path = f"{uuid.uuid4()}.mp4"
    try:
        # First attempt with hardware acceleration
        command = [
            'ffmpeg',
            '-y',
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
            '-i', input_path,
            '-c:v', 'h264_nvenc',  # Switch to H.264 for faster encoding
            '-preset', 'p1',  # Faster preset
            '-rc:v', 'vbr',
            '-cq', '26',  # Higher CQ value for speed
            '-b:v', '4M',  # Slightly lower bitrate
            '-maxrate', '8M',
            '-bufsize', '8M',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            output_path
        ]

        # Run ffmpeg with a timeout
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=300)

        print(f"FFmpeg output: {stdout.decode() if stdout else ''}")
        print(f"FFmpeg error: {stderr.decode() if stderr else ''}")

        if process.returncode != 0:
            # If hardware encoding fails, try software encoding
            print("Hardware encoding failed, falling back to software encoding")
            fallback_command = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-c:v', 'libx264',  # Use software H.264 encoder
                '-preset', 'fast',  # Fast preset for reasonable speed
                '-crf', '23',  # Constant rate factor for quality
                '-c:a', 'copy',  # Copy audio
                '-movflags', '+faststart',
                output_path
            ]

            process = subprocess.Popen(
                fallback_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=300)

            if process.returncode != 0:
                print(f"FFmpeg output: {stdout.decode() if stdout else ''}")
                print(f"FFmpeg error: {stderr.decode() if stderr else ''}")
                raise subprocess.CalledProcessError(process.returncode, fallback_command, stderr)

        return output_path

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e}")
        raise CustomException(code="conversion_failed", message="Failed to convert video format")
    except subprocess.TimeoutExpired:
        process.kill()
        raise CustomException(code="conversion_timeout", message="Conversion timed out")