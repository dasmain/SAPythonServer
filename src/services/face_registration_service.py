import os
import cv2
import pymongo
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
import torch

# load_dotenv()

# mongo_uri = os.getenv("MONGO_URI")
# if not mongo_uri:
#     raise ValueError("No MONGO_URI found in environment variables")

client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["sadb"]
collection = db["face_id"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0).astype(np.float32)
    return img

def extract_features_facenet(face):
    face = preprocess_image(face)
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).to(device)
    features = facenet_model(face)
    return features.detach().cpu().numpy().flatten()

def extract_faces_from_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def register_face(video_file, student_id):
    video_file_path = 'temp/temp_video.mp4'
    os.makedirs(os.path.dirname(video_file_path), exist_ok=True)
    video_file.save(video_file_path)
    cap = cv2.VideoCapture(video_file_path)
    registered_faces = []
    threshold = 0.6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bounding_boxes = extract_faces_from_frame(frame)
        for (x, y, w, h) in bounding_boxes:
            face = frame[y:y+h, x:x+w]
            features = extract_features_facenet(face)
            
            if registered_faces:
                similarity_scores = [cosine(reg_face['faceId'], features) for reg_face in registered_faces]
                # jitne kam value, utna different, 1 se kareeb = accurate
                min_score = min(similarity_scores, default=1.0)
                if min_score < threshold:
                    reg_face_index = similarity_scores.index(min_score)
                    registered_faces[reg_face_index]['faceId'] = features.tolist()
                    registered_faces[reg_face_index]['created_on'] = datetime.now()
                    continue

            registered_faces.append({'faceId': features.tolist(), 'studentId': student_id, 'created_on': datetime.now(), 'deleted_on': None})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if registered_faces:
        collection.update_one({'studentId': student_id}, {'$set': {'faceId': [face['faceId'] for face in registered_faces], 'created_on': datetime.now()}}, upsert=True)
    
    os.remove(video_file_path)