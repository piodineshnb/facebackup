from utils.firebase_initialization import get_db_by_project_id
from utils.custom_exception import CustomException


def get_doc_data(collection, document_id, subcollection=None, sub_document_id=None, project_id="default"):
	db = get_db_by_project_id(project_id)
	print('get_doc_data')
	doc_ref = db.collection(collection).document(document_id).collection(subcollection).document(sub_document_id)
	doc = doc_ref.get()
	if not doc.exists:
		print('No document found with this uid in global.')
		raise CustomException(code="no_document_found",message='No swap session found')
	return doc.to_dict()
