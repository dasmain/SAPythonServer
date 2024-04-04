from flask import Flask
from controllers.face_registration_controller import register_faces

app = Flask(__name__)

# Routes
app.add_url_rule('/face-id/create', '/face-id/create', register_faces, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)