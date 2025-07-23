import subprocess


def check_video_codec(file_path):
    """Check if video uses AV1 codec"""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        codec = result.stdout.strip()
        print(f"Video codec: {codec}")
        return codec == 'av1'
    except subprocess.CalledProcessError:
        return False