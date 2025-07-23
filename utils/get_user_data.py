from utils.firebase_initialization import get_db_by_project_id


def get_user_data(uid, project_id):
    db = get_db_by_project_id(project_id)
    user_ref = db.collection('users').document(uid)
    user_doc = user_ref.get()
    user_data = user_doc.to_dict()
    return user_data, user_doc
