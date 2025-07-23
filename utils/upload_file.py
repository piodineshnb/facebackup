import os
import boto3
from boto3.exceptions import S3UploadFailedError
import time
import random
import re

os.environ['AWS_ACCESS_KEY_ID'] = "00578bf03ff616a000000000a"
os.environ['AWS_SECRET_ACCESS_KEY'] = "K005ZZviU/MgAdlYaPs/zYKpcEkqddg"

session = boto3.Session(
	aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
	aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
	region_name='us-east-005'
)
bucket_name = os.environ.get('BUCKET_NAME','deep-swapper')
s3 = session.client('s3', endpoint_url='https://s3.us-east-005.backblazeb2.com')


def upload_file(file_path: str, file_name: str, file_extension: str, max_retries=5, project_id='default') -> str:
    """
    Uploads file to storage
    :param file_path: path to file
    :return: url to uploaded file
    """
    # Determine the effective bucket name based on project_id
    effective_bucket_name = "user-content-fs" if project_id == "remixAi" else bucket_name

    if file_extension == ".gif":
        file_name = f"swaps/gif/{file_name}"
    elif file_extension == ".webm":
        file_name = f"preview/video/{file_name}"
    elif file_extension == "thumbnail":
        file_name = f"preview/thumbnail/{file_name}"
    else:
        file_name = f"swaps/video/{file_name}"

    print(f"Uploading {file_name} to {effective_bucket_name}")
    for i in range(max_retries):
        try:
            s3.upload_file(file_path, effective_bucket_name, file_name)
            prefixes = {
                'maxStudio': "https://cdn.maxstudio.ai/file/",
                'remixAi': "https://cdn.remixai.io/file/",
            }
            prefix = prefixes.get(project_id, "https://cdn.deepswapper.com/file/")
            return f"{prefix}{effective_bucket_name}/{file_name}"
        except S3UploadFailedError as e:
            print(f"Upload failed on attempt {i + 1}: {e}")
            time.sleep((i + 1) * random.uniform(1, 2))  # increasing delay with some randomness

    raise Exception("Upload failed after maximum retries")

