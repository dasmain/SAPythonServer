import os
import cv2
import numpy as np
import pymongo
from scipy.spatial.distance import cosine
from src.services.get_faceId import get_all_faceIds
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

# Function to extract embeddings using the loaded model
def extract_embeddings(image):
    # Preprocess input image
    preprocessed_image = preprocess_image(image)
    # Perform inference (prediction) using the loaded model
    embeddings = model.predict(np.expand_dims(preprocessed_image, axis=0))
    return embeddings.flatten()

def recognize_face(image):
    # Extract embeddings for the provided image
    image_embeddings = extract_embeddings(image)
    
    # Initialize list to store similarity scores for each student
    recognized_ids = []
    threshold = 0.45

    # Iterate over each student's face IDs
    for student_id, student_face_ids in registered_embeddings.items():
        # Calculate similarity scores between image embeddings and each student's face IDs
        student_similarity_scores = [cosine(face_id, image_embeddings) for face_id in student_face_ids]
        
        # Find the minimum similarity score for this student
        min_similarity_score = min(student_similarity_scores)
        
        # Append the minimum similarity score to the list
        #similarity_scores.append(min_similarity_score)

        if min_similarity_score < threshold:
            recognized_ids.append(student_id)
    
    return recognized_ids

def detect_and_recognize_faces(image):
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    recognized_faces = []

    # Process each detected face separately
    for (x, y, w, h) in faces:
        # Extract the face region from the image
        face = image[y:y+h, x:x+w]
        # Perform face recognition for the extracted face
        recognized_faces = recognize_face(face)
    
    return recognized_faces


def recognition_service(image_file):
    image_file_path = 'temp/temp_image.jpg'
    os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
    image_file.save(image_file_path)
    image = cv2.imread(image_file_path)
    recognized_faces = detect_and_recognize_faces(image)  # Detect and recognize faces in the image
    print(recognized_faces)

    os.remove(image_file_path)
    return recognized_faces