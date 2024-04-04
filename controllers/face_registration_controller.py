from flask import request, jsonify
from services.face_registration_service import register_face

def register_faces():
    # Check if video file is provided
    if 'faceId' not in request.files:
        return jsonify({'success': False, 'data': {}, 'message': 'No video file provided'}), 400

    # Get form data
    student_id = request.form.get('studentId')
    face_id = request.files['faceId']

    # Check if faceId and studentId are provided
    if not face_id or not student_id:
        return jsonify({'success': False, 'data': {}, 'message': 'Missing faceId or studentId in form data'}), 400

    # Call function to register faces from video
    register_face(face_id, student_id)

    return jsonify({'success': True, 'data': {}, 'message': 'Face registration completed'}), 200
