import os
import uuid
import requests
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import subprocess
import threading
import time

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
from utils.update_progress import ProgressUpdater, update_progress
from utils.update_swap_status_local import update_swap_status_local
from utils.custom_exception import CustomException
from facefusion.processors.core import set_total_faces, read_progress_tempfile
from facefusion.core import process_swap_job, execute_job

from cachetools import TTLCache
import uvicorn
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# CORS Config
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = TTLCache(maxsize=10, ttl=172800)
job_lock = threading.Lock()
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Global flag
enable_face_enhance = True

@app.get('/')
def home():
    return "Face Swap API is running."

@app.post("/set_face_enhance")
async def set_face_enhance(enable: bool):
    global enable_face_enhance
    enable_face_enhance = enable
    return {"enable_face_enhance": enable_face_enhance}

@app.post("/swap")
def swap(params: dict, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    swap_doc_id = params.get("swapId")
    if swap_doc_id:
        progress_cache[swap_doc_id] = 0
    jobs[job_id] = {
        "status": "IN_PROGRESS",
        "swapId": swap_doc_id,
        "userId": params.get("userId"),
        "projectId": params.get("projectId", "default"),
        "progress": 0
    }
    set_total_faces(len(params["faces"]))
    background_tasks.add_task(execute_job, job_id, params, jobs)
    return {"jobId": job_id}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id in jobs:
        job = jobs[job_id]
        progress = read_progress_tempfile()
        return {
            **job,
            "progress": progress
        }
    return {"error": {"code": 404, "message": "Job not found"}}

@app.get("/status")
def get_latest_job_status():
    if jobs:
        last_job_id = list(jobs.keys())[-1]
        return get_job_status(last_job_id)
    return {"status": "idle"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
