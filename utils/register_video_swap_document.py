from utils.update_document import update_document
from firebase_admin import firestore
from utils.get_user_data import get_user_data


def register_video_swap_document(user_id, swap_doc_id, faces,project_id):
	print("Registering video swap in the database")
	user_data, user_doc = get_user_data(user_id,project_id)
	if user_data is None:
		is_history_disabled = False
	else:
		is_history_disabled = user_data.get('isHistoryDisabled', False)
	obj = {'updatedAt': firestore.firestore.SERVER_TIMESTAMP, 'status': 'processing',
		   'isHistoryDisabled': is_history_disabled,
		   'input.faces': faces
		   }
	doc_id = update_document(collection="users", document_id=user_id, document_obj=obj, subcollection="swaps",
							 sub_document_id=swap_doc_id,project_id=project_id)
	print(f"updated document with ID: {doc_id}")
	return doc_id
