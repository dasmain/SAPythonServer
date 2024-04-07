from flask import Flask
from src.controllers.face_registration_controller import register_faces
from src.controllers.models_controller import create_model

app = Flask(__name__)

# Routes
app.add_url_rule('/api/v1/face-id/create', '/api/v1/face-id/create', register_faces, methods=['POST'])
app.add_url_rule('/api/v1/face-id/model/face-recognition', '/api/v1/face-id/model/face-recognition', create_model, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)