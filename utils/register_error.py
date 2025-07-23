from firebase_admin import firestore
from utils.update_document import update_document


def register_error(user_id, swap_doc_id, obj,project_id):
	try:
		obj['status'] = 'failed'
		update_document(collection="users", document_id=user_id, document_obj=obj, subcollection="swaps",
					sub_document_id=swap_doc_id,project_id=project_id)
	except Exception as e:
		print("Error in register_error: ", str(e))
