from flask import request, jsonify
from src.models.face_recognition_model import build_and_train_model

def create_model():

    build_and_train_model('../data/train', '../data/validation')

    return jsonify({'success': True, 'data': {}, 'message': 'Model Creation Complete'}), 200
