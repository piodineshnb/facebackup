import mimetypes


def is_video_check(file_path: str,target_extension) -> bool:
    if target_extension == ".quicktime":
        return True
    """Check if the file is a video file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type.split("/")[0] == "video"
    return False
