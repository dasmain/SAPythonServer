from flask import Flask
from src.controllers.face_registration_controller import register_faces
from src.controllers.face_recognition_controller import recognize_faces

app = Flask(__name__)

# Routes
app.add_url_rule('/api/v1/face-id/create', '/api/v1/face-id/create', register_faces, methods=['POST'])
app.add_url_rule('/api/v1/face-id/recognize', '/api/v1/face-id/recognize', recognize_faces, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)