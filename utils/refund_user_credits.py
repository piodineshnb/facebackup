from firebase_admin import firestore
from utils.firebase_initialization import get_db_by_project_id


def refund_user_credits(user_id, credits_on_hold, project_id):
	db = get_db_by_project_id(project_id)
	print("refunding credit to user")
	print("user_id", user_id)
	print("credits_on_hold", credits_on_hold)
	user_ref = db.collection('secureUsers').document(user_id)
	user_ref.update({'creditsOnHold': firestore.firestore.Increment(-credits_on_hold),
					 'credits': firestore.firestore.Increment(credits_on_hold)})
	print("Credit refunded successfully")
	return
