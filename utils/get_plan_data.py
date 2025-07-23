from utils.firebase_initialization import get_db_by_project_id


def get_plan_data(project_id):
	db = get_db_by_project_id(project_id)
	plans_ref = db.collection('global').document('plans')
	plans_doc = plans_ref.get()
	plans_data = plans_doc.to_dict()
	return plans_data
