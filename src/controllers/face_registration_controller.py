from flask import request, jsonify
from src.services.face_registration_service import register_face

def register_faces():
    if 'faceId' not in request.files:
        return jsonify({'success': False, 'data': {}, 'message': 'No video file provided'}), 400

    student_id = request.form.get('studentId')
    face_id = request.files['faceId']

    if not face_id or not student_id:
        return jsonify({'success': False, 'data': {}, 'message': 'Missing faceId or studentId in form data'}), 400
    
    #get faceid by student_id to check if it alr exists or not

    register_face(face_id, student_id)

    return jsonify({'success': True, 'data': {}, 'message': 'Face registration completed'}), 200
