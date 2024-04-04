from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from scipy.spatial.distance import cosine
import pymongo
from datetime import datetime
from bson import ObjectId

# MongoDB connection
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["sadb"]
collection = db["face_id"]

# Load pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess an image for VGG16 model
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

# Function to extract features using VGG16 model
def extract_features_vgg16(face):
    face = preprocess_image(face)
    face = np.expand_dims(face, axis=0)
    features = vgg16_model.predict(face)
    return features.flatten()

# Function to extract faces from video frames
def extract_faces_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def register_face(video_file, student_id):
    video_file_path = 'temp/temp_video.mp4'  # Define the path where you want to save the file
    video_file.save(video_file_path)
    cap = cv2.VideoCapture(video_file_path)
    registered_faces = []  # Initialize with empty list for each video
    threshold = 0.6  # Example threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces_from_frame(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            features = extract_features_vgg16(face)
            
            # Check if any registered faces exist
            if registered_faces:
                # Compare extracted features with registered faces
                similarity_scores = [cosine(reg_face['faceId'], features) for reg_face in registered_faces]
                min_score = min(similarity_scores, default=1.0)  # Set default value to 1.0 if list is empty
                if min_score < threshold:
                    # Face is already registered
                    continue

            # Register new face
            registered_faces.append({'faceId': features.tolist(), 'studentId': student_id, 'created_on': datetime.now(), 'deleted_on': None})
            # Optionally, save the registered face to your dataset
            # Optionally, update your facial recognition model

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save registered faces to MongoDB
    if registered_faces:
        collection.insert_many(registered_faces)
