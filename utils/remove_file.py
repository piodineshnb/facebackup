import os
def remove_file(file_path):
	try:
		os.remove(file_path)
	except Exception as e:
		print(f"Error removing file: {file_path}")
