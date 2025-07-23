from PIL import Image
from utils.custom_exception import CustomException
import mimetypes
mimetypes.add_type('image/webp', '.webp')


def check_animated_image_type(file_path):
    """Determine the image type of the file using Pillow"""
    print("file path", file_path)
    try:
        with Image.open(file_path) as img:
            img_format = img.format.lower()
            if img_format not in ['gif', 'webp']:
                mime_type, _ = mimetypes.guess_type(file_path, strict=False)
                if mime_type:
                    return mime_type.split('/')[1]
                return None
            return img_format
    except IOError:
        raise CustomException(code="invalid_animated_image", message="Invalid animated image file")
