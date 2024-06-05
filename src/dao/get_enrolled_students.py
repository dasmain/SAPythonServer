import pymongo
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("No MONGO_URI found in environment variables")

client = pymongo.MongoClient(mongo_uri)
db = client["sadb"]
collection = db["course"]

def get_all_students(course_id):
    student_ids = []
    course_id_obj = ObjectId(course_id)
    cursor = collection.find({"_id": course_id_obj}, {"studentsEnrolled": 1})

    for doc in cursor:
        student_ids.extend(doc["studentsEnrolled"])

    return student_ids

