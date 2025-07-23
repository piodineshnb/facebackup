import os
import json
import base64
from firebase_admin import credentials, firestore, initialize_app

# Retrieve the base64 encoded JSON string from an environment variable
encoded_json_cred = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
if encoded_json_cred is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON' not found")

encoded_json_cred_web_swap_my_face = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_WEB_SWAP_MY_FACE')
if encoded_json_cred_web_swap_my_face is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_WEB_SWAP_MY_FACE' not found")

encoded_json_cred_mobile_rogue = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_MOBILE_ROGUE')
if encoded_json_cred_mobile_rogue is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_MOBILE_ROGUE' not found")

encoded_json_cred_mobile_syndicate = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_MOBILE_SYNDICATE')
if encoded_json_cred_mobile_syndicate is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_MOBILE_SYNDICATE' not found")

encoded_json_cred_web_max_studio = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_WEB_MAX_STUDIO')
if encoded_json_cred_web_max_studio is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_WEB_MAX_STUDIO' not found")

encoded_json_cred_max_studio_apis = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_MAX_STUDIO_APIS')
if encoded_json_cred_max_studio_apis is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_MAX_STUDIO_APIS' not found")

encoded_json_cred_mobile_odyssey = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_MOBILE_ODYSSEY')
if encoded_json_cred_mobile_odyssey is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_MOBILE_ODYSSEY' not found")

encoded_json_cred_web_remix_ai = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_WEB_REMIX_AI')
if encoded_json_cred_web_remix_ai is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_WEB_REMIX_AI' not found")\

encoded_json_cred_web_chroma_studio = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_WEB_CHROMA_STUDIO')
if encoded_json_cred_web_chroma_studio is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_WEB_CHROMA_STUDIO' not found")

encoded_json_cred_test_dinesh = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_TEST_DINESH')
if encoded_json_cred_test_dinesh is None:
    raise ValueError("Environment variable 'FIREBASE_SERVICE_ACCOUNT_JSON_TEST_DINESH' not found")





# Decode the base64 string
decoded_json_cred = base64.b64decode(encoded_json_cred)
decoded_json_cred_web_swap_my_face = base64.b64decode(encoded_json_cred_web_swap_my_face)
decoded_json_cred_2 = base64.b64decode(encoded_json_cred_mobile_rogue)
decoded_json_cred_syndicate = base64.b64decode(encoded_json_cred_mobile_syndicate)
decoded_json_cred_web_max_studio = base64.b64decode(encoded_json_cred_web_max_studio)
decoded_json_cred_max_studio_apis = base64.b64decode(encoded_json_cred_max_studio_apis)
decoded_json_cred_mobile_odyssey = base64.b64decode(encoded_json_cred_mobile_odyssey)
decoded_json_cred_web_remix_ai = base64.b64decode(encoded_json_cred_web_remix_ai)
decoded_json_cred_web_chroma_studio = base64.b64decode(encoded_json_cred_web_chroma_studio)
decoded_json_cred_test_dinesh = base64.b64decode(encoded_json_cred_test_dinesh)

# Convert the decoded JSON string into a dictionary
key_dict = json.loads(decoded_json_cred)
key_dict_web_swap_my_face = json.loads(decoded_json_cred_web_swap_my_face)
key_dict_mobile_rogue = json.loads(decoded_json_cred_2)
key_dict_mobile_syndicate = json.loads(decoded_json_cred_syndicate)
key_dict_web_max_studio = json.loads(decoded_json_cred_web_max_studio)
key_dict_max_studio_apis = json.loads(decoded_json_cred_max_studio_apis)
key_dict_mobile_odyssey = json.loads(decoded_json_cred_mobile_odyssey)
key_dict_web_remix_ai = json.loads(decoded_json_cred_web_remix_ai)
key_dict_web_chroma_studio = json.loads(decoded_json_cred_web_chroma_studio)
key_dict_test_dinesh = json.loads(decoded_json_cred_test_dinesh)

# Use the service account
cred = credentials.Certificate(key_dict)
cred_web_swap_my_face = credentials.Certificate(key_dict_web_swap_my_face)
cred_mobile_rogue = credentials.Certificate(key_dict_mobile_rogue)
cred_mobile_syndicate = credentials.Certificate(key_dict_mobile_syndicate)
cred_web_max_studio = credentials.Certificate(key_dict_web_max_studio)
cred_max_studio_apis = credentials.Certificate(key_dict_max_studio_apis)
cred_mobile_odyssey = credentials.Certificate(key_dict_mobile_odyssey)
cred_web_remix_ai = credentials.Certificate(key_dict_web_remix_ai)
cred_web_chroma_studio = credentials.Certificate(key_dict_web_chroma_studio)
cred_test_dinesh = credentials.Certificate(key_dict_test_dinesh)

app = initialize_app(cred, name='firebase')
app_web_swap_my_face = initialize_app(cred_web_swap_my_face, name='firebase_swap_my_face')
app_mobile_rogue = initialize_app(cred_mobile_rogue, name='firebase2')
app_mobile_syndicate = initialize_app(cred_mobile_syndicate, name='firebase_syndicate')
app_web_max_studio = initialize_app(cred_web_max_studio, name='firebase_max_studio')
app_max_studio_apis = initialize_app(cred_max_studio_apis, name='firebase_apis')
app_mobile_odyssey = initialize_app(cred_mobile_odyssey, name='firebase_odyssey')
app_web_remix_ai = initialize_app(cred_web_remix_ai, name='firebase_remix_ai')
app_web_chroma_studio = initialize_app(cred_web_chroma_studio, name='firebase_web_chroma_studio')
app_test_dinesh = initialize_app(cred_test_dinesh, name='firebase_test_dinesh')

db_web = firestore.client(app=app)
db_web_swap_my_face = firestore.client(app=app_web_swap_my_face)
db_rogue = firestore.client(app=app_mobile_rogue)
db_syndicate = firestore.client(app=app_mobile_syndicate)
db_max_studio = firestore.client(app=app_web_max_studio)
db_apis = firestore.client(app=app_max_studio_apis)
db_odyssey = firestore.client(app=app_mobile_odyssey)
db_remix_ai = firestore.client(app=app_web_remix_ai)
db_web_chroma_studio = firestore.client(app=app_web_chroma_studio)
db_test_dinesh = firestore.client(app=app_test_dinesh)


def get_db_by_project_id(project_id):
    switcher = {
        "default": db_web,
        "swapMyFace": db_web_swap_my_face,
        "rogue": db_rogue,
        "syndicate": db_syndicate,
        "maxStudio": db_max_studio,
        "apiMaxStudio": db_apis,
        "odyssey": db_odyssey,
        "remixAi": db_remix_ai,
        "chromaStudio": db_web_chroma_studio,
        "testDinesh": db_test_dinesh,
    }
    return switcher.get(project_id, db_web)

