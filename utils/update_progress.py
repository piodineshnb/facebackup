from utils.update_document import update_document
import time
from tqdm import tqdm
from cachetools import TTLCache
import facefusion.globals
import runpod
from datetime import datetime, timezone
progress_cache = TTLCache(maxsize=10, ttl=172800)


class ProgressUpdater:
	def __init__(self, update_interval, min_percentage_increment, user_id, session_id):
		self.total = None
		self.update_interval = update_interval
		self.min_percentage_increment = min_percentage_increment
		self.user_id = user_id
		self.session_id = session_id
		self.last_update_time = time.time()
		self.last_progress_percentage = 0

	def set_total(self, total):
		self.total = total

	def update_progress_firebase(self, progress, set_hundred=False,project_id="default"):
		current_time = time.time()
		if set_hundred:
			progress_percentage = 100
		else:
			progress_percentage = int((progress / self.total) * 100)

		if not set_hundred and progress_percentage == 100:
			return

		if (current_time - self.last_update_time >= self.update_interval or
			progress_percentage - self.last_progress_percentage >= self.min_percentage_increment or
			set_hundred):
			tqdm.write(f"Updating progress in Firestore: {progress_percentage}%")
			tqdm.write(f"current_time - last_update_time: {current_time - self.last_update_time}")
			tqdm.write(
				f"progress_percentage - last_progress_percentage: {progress_percentage - self.last_progress_percentage}")

			obj = {
				"progress": progress_percentage
			}
			self.last_update_time = current_time
			self.last_progress_percentage = progress_percentage
			# Update Firestore document
			tqdm.write(f"Updating progress in Firestore: {progress_percentage}%")
			update_document(collection="users", document_id=self.user_id, document_obj=obj, subcollection="swaps",
							sub_document_id=self.session_id,project_id=project_id)

	def update_progress_local(self, progress, set_hundred=False):
		if set_hundred:
			progress_percentage = 100
		else:
			progress_percentage = int((progress / self.total) * 100)

		if set_hundred or progress_percentage != 100:
			progress_cache[self.session_id] = progress_percentage
			now = datetime.now(timezone.utc)

			# Convert to milliseconds since the epoch
			milliseconds = int(now.timestamp() * 1000)
			facefusion.globals.last_updated = milliseconds

	def update_progress_serverless(self, progress, set_hundred=False):
		if set_hundred:
			progress_percentage = 100
		else:
			progress_percentage = int((progress / self.total) * 100)

		if not set_hundred:
			# runpod.serverless.progress_update(facefusion.globals.job_input, f"{progress_percentage}")
			pass


def update_progress(progress : int | float , progress_updater: ProgressUpdater, set_hundred=False,project_id="default"):
	progress_updater.update_progress_firebase(progress=progress, set_hundred=set_hundred,project_id=project_id)
	progress_updater.update_progress_local(progress=progress, set_hundred=set_hundred)
	progress_updater.update_progress_serverless(progress=progress, set_hundred=set_hundred)
