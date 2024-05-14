import pymongo
from bson import ObjectId

# MongoDB connection
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["sadb"]
collection = db["course"]

def get_all_students(course_id):
    student_ids = []
    course_id_obj = ObjectId(course_id)
    cursor = collection.find({"_id": course_id_obj}, {"studentsEnrolled": 1})

    for doc in cursor:
        student_ids.extend(doc["studentsEnrolled"])

    return student_ids

