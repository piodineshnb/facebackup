from utils.firebase_initialization import get_db_by_project_id


def create_document(collection, document_id, document_obj, subcollection=None, project_id):

    db = get_db_by_project_id(project_id)

    doc_ref = db.collection(collection).document(document_id)
    if subcollection:
        doc_ref = doc_ref.collection(subcollection).document()
    doc_ref.set(document_obj)
    return doc_ref.id
