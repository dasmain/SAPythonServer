# import os
# import cv2
# import numpy as np
# import pymongo
# from dotenv import load_dotenv
# from scipy.spatial.distance import cosine
# from src.dao.get_faceId import get_all_faceIds
# from src.dao.get_enrolled_students import get_all_students
# from src.services.niqaab_recognition_service import niqab_detection
# from tensorflow.keras.models import load_model # type: ignore
# from tensorflow.keras.applications.mobilenet import preprocess_input # type: ignore
# from tensorflow.keras.applications.mobilenet import MobileNet # type: ignore
# #from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2 # type: ignore


# load_dotenv()

# # MongoDB connection
# mongo_uri = os.getenv("MONGO_URI")
# if not mongo_uri:
#     raise ValueError("No MONGO_URI found in environment variables")

# client = pymongo.MongoClient(mongo_uri)
# db = client["sadb"]
# collection = db["face_id"]

# # Load the saved model
# #model = load_model('trained_model/custom_face_recognition_model.h5')
# model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# #model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Load the saved embeddings of registered faces
# registered_embeddings = get_all_faceIds()

# def preprocess_image(img):
#     img = cv2.resize(img, (224, 224))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = preprocess_input(img)
#     return img

# def extract_embeddings(image):
#     preprocessed_image = preprocess_image(image)
#     embeddings = model.predict(np.expand_dims(preprocessed_image, axis=0))
#     return embeddings.flatten()

# def recognize_face(image, recognized):
#     image_embeddings = extract_embeddings(image)
    
#     recognized_ids = []
#     min_similarity_object = []
#     threshold = 0.61
#     min_score = None
#     min_student_id = None

#     for student_id, student_face_ids in registered_embeddings.items():
#         #if student_id in recognized:
#             #continue

#         student_similarity_scores = [cosine(face_id, image_embeddings) for face_id in student_face_ids]
#         print("Student Similarity Scores:" + str(student_id) + ": " + str(student_similarity_scores))
#         min_similarity_score = min(student_similarity_scores)

#         if min_similarity_score < threshold:
#             min_similarity_object.append({"student_id": student_id, "min_similarity_score": min_similarity_score})
    
#     if min_similarity_object:
#         min_score = min_similarity_object[0]['min_similarity_score']
#         min_student_id = min_similarity_object[0]['student_id']

#     for item in min_similarity_object:
#         if item['min_similarity_score'] < min_score:
#             min_score = item['min_similarity_score']
#             min_student_id = item['student_id']
    
#     print("Added:" + str(min_student_id))
#     recognized_ids.append(min_student_id)
#     return recognized_ids

# # def detect_and_recognize_faces(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
# #     recognized_faces = []

# #     for idx, (x, y, w, h) in enumerate(faces):
# #         face = image[y:y+h, x:x+w]
# #         recognized_face_ids = recognize_face(face, recognized_faces)
# #         recognized_faces.extend(recognized_face_ids)
# #         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
# #         cv2.putText(image, str(idx+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
# #     cv2.imshow("Result", image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #     return recognized_faces, len(faces)

# def detect_and_recognize_faces(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
#     recognized_faces = []

#     for (x, y, w, h) in faces:
#         face = image[y:y+h, x:x+w]
#         recognized_face_ids = recognize_face(face, recognized_faces)
#         recognized_faces.extend(recognized_face_ids)
    
#     return recognized_faces, len(faces)


# def recognition_service(image_file, course_id):
#     image_file_path = 'temp/temp_image.jpg'
#     os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
#     image_file.save(image_file_path)
#     image = cv2.imread(image_file_path)
#     recognized_faces, headcount = detect_and_recognize_faces(image)
#     all_students = get_all_students(course_id)
#     attendance_records = []
#     niqab_count = niqab_detection(image_file_path)

#     for student_id in all_students:
#         if student_id in recognized_faces:
#             attendance_records.append({"studentId": student_id, "status": "present"})
#         else:
#             attendance_records.append({"studentId": student_id, "status": "absent"})

#     os.remove(image_file_path)
#     return attendance_records, headcount, niqab_count

import os
import cv2
import numpy as np
import pymongo
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
from src.dao.get_faceId import get_all_faceIds
from src.dao.get_enrolled_students import get_all_students
from src.services.niqaab_recognition_service import niqab_detection
from facenet_pytorch import InceptionResnetV1
import torch
import base64

load_dotenv()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("No MONGO_URI found in environment variables")

client = pymongo.MongoClient(mongo_uri)
db = client["sadb"]
collection = db["face_id"]

# Load the FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the saved embeddings of registered faces
registered_embeddings = get_all_faceIds()

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0).astype(np.float32)
    return img

def extract_embeddings(image):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = torch.tensor(preprocessed_image).permute(2, 0, 1).unsqueeze(0).to(device)
    embeddings = model(preprocessed_image)
    return embeddings.detach().cpu().numpy().flatten()

def recognize_face(image, recognized):
    image_embeddings = extract_embeddings(image)
    
    recognized_ids = []
    min_similarity_object = []
    threshold = 0.47
    min_score = None
    min_student_id = None

    for student_id, student_face_ids in registered_embeddings.items():
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

# def detect_and_recognize_faces(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
#     recognized_faces = []

#     for (x, y, w, h) in faces:
#         face = image[y:y+h, x:x+w]
#         recognized_face_ids = recognize_face(face, recognized_faces)
#         recognized_faces.extend(recognized_face_ids)
    
#     return recognized_faces, len(faces)

def detect_and_recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    recognized_faces = []
    recognized_amount = 0

    for idx, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        recognized_face_ids = recognize_face(face, recognized_faces)
        recognized_faces.extend(recognized_face_ids)
        if recognized_face_ids == [None]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, str(idx+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        elif recognized_face_ids:
            recognized_amount += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, str(idx+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # cv2.imshow("Result", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return recognized_faces, len(faces), image, recognized_amount

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def recognition_service(image_file, course_id):
    image_file_path = 'temp/temp_image.jpg'
    os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
    image_file.save(image_file_path)
    image = cv2.imread(image_file_path)
    recognized_faces, headcount, new_image, recognized_amount = detect_and_recognize_faces(image)
    unrecognized_amount = headcount - recognized_amount
    base64image = encode_image_to_base64(new_image)
    all_students = get_all_students(course_id)
    attendance_records = []
    niqab_count = niqab_detection(image_file_path)

    for student_id in all_students:
        if student_id in recognized_faces:
            attendance_records.append({"studentId": student_id, "status": "present"})
        else:
            attendance_records.append({"studentId": student_id, "status": "absent"})

    os.remove(image_file_path)
    return attendance_records, headcount, niqab_count, base64image, recognized_amount, unrecognized_amount
