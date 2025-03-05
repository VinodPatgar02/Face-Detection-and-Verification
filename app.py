import os
import cv2
import numpy as np
import torch
import time
import base64
import json
from flask import Flask, jsonify, request, render_template, send_from_directory
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import io
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
from flask_cors import CORS

# Flask App Setup
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("API_KEY")  # Get the API key from the environment variable

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/verify": {"origins": "http://127.0.0.1:5000"}})
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "static/uploads")
app.config['REFERENCE_FOLDER'] = os.getenv("REFERENCE_FOLDER", "reference_images")
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB Max File Size

# Load staff details from JSON file
with open("staff_details.json", "r") as f:
    staff_details = json.load(f)

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REFERENCE_FOLDER'], exist_ok=True)

# Face Detection & Recognition Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Face Preprocessing Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

THRESHOLD = 0.7

# Load Reference Faces
reference_db = {}
for filename in os.listdir(app.config['REFERENCE_FOLDER']):
    if filename.split('.')[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
        staff_name = filename.split('.')[0]  # Use filename (without extension) as staff identifier
        img_path = os.path.join(app.config['REFERENCE_FOLDER'], filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            face = img_rgb[y1:y2, x1:x2]
            face = transform(face).unsqueeze(0).to(device)
            reference_db[staff_name] = resnet(face).detach().cpu().numpy().flatten()

# Secure API Key Check
def require_api_key(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 403  # Forbidden
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models/<path:filename>')
def serve_models(filename):
    return send_from_directory(os.path.join(app.root_path, 'models'), filename)

@app.route('/verify', methods=['POST'])
@require_api_key  # This will enforce API key checking on the verify route
def verify():
    try:
        # Handle direct file uploads
        if 'image' in request.files:
            img_file = request.files['image']
            if not img_file.filename.lower().endswith(tuple(app.config['ALLOWED_EXTENSIONS'])):
                return jsonify({'error': 'Invalid file type. Only JPG, PNG, and JPEG are allowed.'}), 400
            img_bytes = img_file.read()
        else:
            # Handle base64 URL
            img_data = request.form.get('image', '')
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
        
        # Convert bytes to image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return jsonify({'error': 'No face detected in the image'}), 400
        
        # Process the first detected face
        x1, y1, x2, y2 = map(int, boxes[0])
        face = img_rgb[y1:y2, x1:x2]
        
        # Transform and get embedding
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
        
        # Compare with reference database
        matches = []
        for name, ref_embedding in reference_db.items():
            similarity = np.dot(embedding, ref_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)
            )
            if similarity > THRESHOLD:
                matches.append({
                    'name': name,
                    'confidence': float(similarity)
                })
        
        # Sort matches by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        # Add staff details if a match is found
        if matches:
            top_match = matches[0]
            staff_name = top_match['name']

            # Look up staff details
            details = staff_details.get(staff_name, {})
            return jsonify({
                'result': matches,
                'staff_details': details  # Include staff details in response
            })

        return jsonify({'result': [], 'error': 'No match found'}), 200

    except Exception as e:
        app.logger.error(f"Error in /verify: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)