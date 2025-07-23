from utils.firebase_initialization import get_db_by_project_id
from utils.custom_exception import CustomException
from utils.get_doc_data import get_doc_data
from utils.get_plan_data import get_plan_data


def check_and_hold_user_credits(user_id, swap_doc_id,duration_in_seconds, project_id):
	db = get_db_by_project_id(project_id)
	user_ref = db.collection('secureUsers').document(user_id)
	user_doc = user_ref.get()
	if not user_doc.exists:
		print('No document found with this uid in secureUsers.')
		raise CustomException(code='user_not_found', message='User not found')
	user_data = user_doc.to_dict()
	swap_session_data = get_doc_data('users', user_id, 'swaps', swap_doc_id, project_id)
	credits_required = swap_session_data['credits']

	user_available_credits = user_data.get("credits", 0)
	if user_available_credits < credits_required:
		print('User has no credit left')
		raise CustomException(code='insufficient_credits', message='Insufficient credits')

	# plans = get_plan_data(project_id)
	# plans = plans.get('plans', [])
	# subscription = user_data.get("subscription", None)
	# user_plan_id = subscription.get("planId", None) if subscription else None
	# print(f'user_plan_id: {user_plan_id}')
	# print(f'duration_in_seconds: {duration_in_seconds}')
	# if user_plan_id and duration_in_seconds:
	# 	print('Checking if user has exceeded the maximum duration allowed for the current plan')
	# 	for plan in plans:
	# 		if plan['id'] == user_plan_id:
	# 			max_duration_seconds = plan['maxDurationSeconds']
	# 			print(f'max_duration_seconds: {max_duration_seconds}')
	# 			if duration_in_seconds > max_duration_seconds:
	# 				print('User has exceeded the maximum duration allowed for the current plan')
	# 				raise CustomException(code='exceeded_max_duration', message='Exceeded maximum duration allowed for the current plan')

	user_ref.update({
		'credits': user_available_credits - credits_required,
		'creditsOnHold': user_data.get("creditsOnHold", 0) + credits_required
	})
	return credits_required