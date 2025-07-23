from utils.firebase_initialization import get_db_by_project_id
from utils.custom_exception import CustomException


def update_document(collection, document_id, document_obj, subcollection=None, sub_document_id=None,project_id="default"):
    db = get_db_by_project_id(project_id)
    doc_ref = db.collection(collection).document(document_id).collection(subcollection).document(sub_document_id)
    if doc_ref.get().exists:
        doc_ref.update(document_obj)
        return doc_ref.id
    else:
        raise CustomException(code='Swap_Document_Not_Found', message='Swap Document not found')
