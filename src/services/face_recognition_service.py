import os
import cv2
import numpy as np
import pymongo
from scipy.spatial.distance import cosine
from src.dao.get_faceId import get_all_faceIds
from src.dao.get_enrolled_students import get_all_students
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet import preprocess_input # type: ignore
from tensorflow.keras.applications.mobilenet import MobileNet # type: ignore

# MongoDB connection
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["sadb"]
collection = db["face_id"]

# Load the saved model
#model = load_model('trained_model/custom_face_recognition_model.h5')
model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the saved embeddings of registered faces
registered_embeddings = get_all_faceIds()

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

def extract_embeddings(image):
    preprocessed_image = preprocess_image(image)
    embeddings = model.predict(np.expand_dims(preprocessed_image, axis=0))
    return embeddings.flatten()

def recognize_face(image, recognized):
    image_embeddings = extract_embeddings(image)
    
    recognized_ids = []
    min_similarity_object = []
    threshold = 0.61
    min_score = None
    min_student_id = None

    for student_id, student_face_ids in registered_embeddings.items():
        if student_id in recognized:
            continue

        student_similarity_scores = [cosine(face_id, image_embeddings) for face_id in student_face_ids]
        print("Student Similarity Scores:" + str(student_id) + ": " + str(student_similarity_scores))
        min_similarity_score = min(student_similarity_scores)

        if min_similarity_score < threshold:
            min_similarity_object.append({"student_id": student_id, "min_similarity_score": min_similarity_score})
    
    if min_similarity_object:
        min_score = min_similarity_object[0]['min_similarity_score']
        min_student_id = min_similarity_object[0]['student_id']

    for item in min_similarity_object:
        if item['min_similarity_score'] < min_score:
            min_score = item['min_similarity_score']
            min_student_id = item['student_id']
    
    print("Added:" + str(min_student_id))
    recognized_ids.append(min_student_id)
    return recognized_ids

def detect_and_recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    recognized_faces = []

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        recognized_face_ids = recognize_face(face, recognized_faces)
        recognized_faces.extend(recognized_face_ids)
    
    return recognized_faces, len(faces)


def recognition_service(image_file, course_id):
    image_file_path = 'temp/temp_image.jpg'
    os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
    image_file.save(image_file_path)
    image = cv2.imread(image_file_path)
    recognized_faces, headcount = detect_and_recognize_faces(image)
    all_students = get_all_students(course_id)
    attendance_records = []

    for student_id in all_students:
        if student_id in recognized_faces:
            attendance_records.append({"studentId": student_id, "status": "present"})
        else:
            attendance_records.append({"studentId": student_id, "status": "absent"})

    os.remove(image_file_path)
    return attendance_records, headcount