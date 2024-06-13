import pymongo
import numpy as np
import os
from dotenv import load_dotenv

# load_dotenv()

# mongo_uri = os.getenv("MONGO_URI")
# if not mongo_uri:
#     raise ValueError("No MONGO_URI found in environment variables")

client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["sadb"]
collection = db["face_id"]

def get_all_faceIds():
    cursor = collection.find({}, {"studentId": 1, "faceId": 1})
    registered_embeddings = {}

    for doc in cursor:
        student_id = doc["studentId"]
        face_ids = doc["faceId"]

        flattened_ids = [np.array(inner_id).flatten() for inner_id in face_ids]

        if student_id in registered_embeddings:
            registered_embeddings[student_id].extend(flattened_ids)
        else:
            registered_embeddings[student_id] = flattened_ids

    return registered_embeddings