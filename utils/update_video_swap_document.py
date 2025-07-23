from firebase_admin import firestore
from utils.update_document import update_document


def update_video_swap_document(user_id, download_url,swap_doc_id,preview_url,total_time,upload_time,thumbnail_url,project_id):
	print("Updating video swap document")
	obj = {
		'output': {
			'mediaUrl': download_url,
			'createdAt': firestore.firestore.SERVER_TIMESTAMP,
			'previewUrl': preview_url,
			'swapTimeInSeconds': round(total_time, 2),
			'thumbnailUrl': thumbnail_url,
			'uploadTimeInSeconds': round(upload_time, 2)

		},
		'status': 'completed'
	}

	doc_id = update_document(collection="users", document_id=user_id, document_obj=obj, subcollection="swaps",sub_document_id=swap_doc_id,project_id=project_id)
	print(f"updated document with ID: {doc_id}")
	return doc_id

