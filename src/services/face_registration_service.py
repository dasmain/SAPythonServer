import os
import cv2
import pymongo
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.mobilenet import preprocess_input # type: ignore
from tensorflow.keras.applications.mobilenet import MobileNet # type: ignore
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2 # type: ignore

# MongoDB connection
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["sadb"]
collection = db["face_id"]

# Load pre-trained MobileNet model
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess an image for MobileNet model
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

# Function to extract features using MobileNet model
def extract_features_mobilenet(face):
    face = preprocess_image(face)
    face = np.expand_dims(face, axis=0)
    features = mobilenet_model.predict(face)
    return features.flatten()

# Function to extract faces from video frames using Haar Cascade
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
            features = extract_features_mobilenet(face)
            
            # Check if any registered faces exist
            if registered_faces:
                # Compare extracted features with registered faces
                similarity_scores = [cosine(reg_face['faceId'], features) for reg_face in registered_faces]
                min_score = min(similarity_scores, default=1.0)
                if min_score < threshold:
                    # Face is already registered, update existing record
                    reg_face_index = similarity_scores.index(min_score)
                    registered_faces[reg_face_index]['faceId'] = features.tolist()
                    registered_faces[reg_face_index]['created_on'] = datetime.now()
                    continue

            # Register new face
            registered_faces.append({'faceId': features.tolist(), 'studentId': student_id, 'created_on': datetime.now(), 'deleted_on': None})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Update MongoDB with the consolidated record
    if registered_faces:
        collection.update_one({'studentId': student_id}, {'$set': {'faceId': [face['faceId'] for face in registered_faces], 'created_on': datetime.now()}}, upsert=True)
    
    os.remove(video_file_path)
