from flask import request, jsonify
from src.services.face_recognition_service import recognition_service
import numpy as np
import cv2

def recognize_faces():
    if 'faceId' not in request.files:
        return jsonify({'success': False, 'data': {}, 'message': 'No image file provided'}), 400

    uploaded_file = request.files['faceId']

    if not uploaded_file:
        return jsonify({'success': False, 'data': {}, 'message': 'No faceId provided in form data'}), 400
    
    data = recognition_service(uploaded_file)

    return jsonify({'success': True, 'data': data, 'message': 'Face recognition completed'}), 200
