from deepface import DeepFace
from deepface.modules.verification import find_distance
import cv2
import time
import pickle
import os
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
import threading
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
EMBEDDINGS_PATH = "./embeddings/embs_facenet512.pkl"
MODEL_NAME = "Facenet512"
DISTANCE_THRESHOLD = 0.78
CONFIDENCE_THRESHOLD = 0.5
INPUT_DIR = "./data"
CROPPED_DIR = "./cropped_faces"
EMBEDDINGS_DIR = "./embeddings"

# Global variables
embs = {}
setup_complete = False

def crop_faces(input_dir, output_dir, detector_backend="opencv"):
    """Crop faces from images using optimized function"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Cropping faces from '{input_dir}' to '{output_dir}'...")
        
        image_files = [f for f in os.listdir(input_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            logger.warning(f"No image files found in '{input_dir}'")
            return False
        
        success_count = 0
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            img_name = img_file.split(".")[0]
            
            try:
                face_objs = DeepFace.extract_faces(
                    img_path,
                    detector_backend=detector_backend,
                    enforce_detection=True,
                    align=True
                )
               
                if len(face_objs) > 0:
                    face = face_objs[0]["face"]
                   
                    # Convert to uint8 if needed
                    if face.dtype == np.float64:
                        face = (face * 255).astype(np.uint8)
                   
                    # Resize to target size
                    face = cv2.resize(face, (224, 224))
                   
                    # Save the face
                    output_path = os.path.join(output_dir, f"{img_name}.jpg")
                    cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    success_count += 1
                   
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
        
        logger.info(f"Successfully cropped {success_count}/{len(image_files)} faces")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error in crop_faces: {str(e)}")
        return False

def extract_embeddings(cropped_faces_dir, output_dir, model_name="Facenet512"):
    """Extract face embeddings from cropped face images"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        embeddings = {}
        
        if not os.path.exists(cropped_faces_dir):
            logger.error(f"Cropped faces directory '{cropped_faces_dir}' not found!")
            return False
        
        image_files = [f for f in os.listdir(cropped_faces_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            logger.error(f"No image files found in '{cropped_faces_dir}'")
            return False
        
        logger.info(f"Extracting embeddings from {len(image_files)} images...")
        
        success_count = 0
        for img_file in image_files:
            img_path = os.path.join(cropped_faces_dir, img_file)
            person_name = os.path.splitext(img_file)[0]
            
            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip"
                )[0]["embedding"]
                
                embeddings[person_name] = embedding
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
        
        if embeddings:
            output_path = os.path.join(output_dir, f"embs_{model_name.lower()}.pkl")
            
            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)
            
            logger.info(f"Successfully saved {len(embeddings)} embeddings to: {output_path}")
            logger.info(f"Registered faces: {list(embeddings.keys())}")
            return True
        else:
            logger.error("No embeddings were extracted!")
            return False
            
    except Exception as e:
        logger.error(f"Error in extract_embeddings: {str(e)}")
        return False

def setup_face_recognition():
    """Complete setup process for face recognition"""
    global setup_complete
    
    try:
        logger.info("Starting Face Recognition Setup...")
        
        # Check if data directory exists
        if not os.path.exists(INPUT_DIR):
            logger.error(f"Input directory '{INPUT_DIR}' not found!")
            os.makedirs(INPUT_DIR, exist_ok=True)
            return False
        
        # Check if there are images in data directory
        image_files = [f for f in os.listdir(INPUT_DIR) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            logger.warning(f"No images found in '{INPUT_DIR}' directory!")
            return False
        
        logger.info(f"Found {len(image_files)} images in data directory")
        
        # Step 1: Crop faces
        if not crop_faces(INPUT_DIR, CROPPED_DIR):
            return False
        
        # Step 2: Extract embeddings
        if not extract_embeddings(CROPPED_DIR, EMBEDDINGS_DIR, MODEL_NAME):
            return False
        
        logger.info("Face Recognition Setup Complete!")
        setup_complete = True
        return True
        
    except Exception as e:
        logger.error(f"Error in setup_face_recognition: {str(e)}")
        return False

def load_embeddings():
    """Load face embeddings from file"""
    global embs
    try:
        with open(EMBEDDINGS_PATH, "rb") as file:
            embs = pickle.load(file)
            logger.info(f"Loaded {len(embs)} face embeddings")
            return True
    except FileNotFoundError:
        logger.warning(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        return False
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return False

def process_frame(frame):
    """Process frame and detect faces"""
    try:
        # Use opencv detector for better performance
        results = DeepFace.extract_faces(
            frame, 
            detector_backend="opencv", 
            enforce_detection=False
        )
        
        detected_faces = []
        
        for result in results:
            if result["confidence"] >= CONFIDENCE_THRESHOLD:
                area = result["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                
                try:
                    embedding = DeepFace.represent(
                        img_path=face_roi,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        detector_backend="skip"
                    )[0]["embedding"]
                    
                    best_match = "Unknown"
                    min_distance = float("inf")
                    
                    for name, known_embedding in embs.items():
                        distance = find_distance(embedding, known_embedding, "euclidean_l2")
                        if distance < min_distance:
                            min_distance = distance
                            best_match = name
                    
                    if min_distance < DISTANCE_THRESHOLD:
                        label = f"{best_match}"
                        confidence = max(0, 1 - (min_distance / DISTANCE_THRESHOLD))
                    else:
                        label = "Unknown"
                        confidence = 0.3
                    
                    detected_faces.append({
                        'name': label,
                        'coordinates': [x, y, x+w, y+h],
                        'confidence': confidence,
                        'distance': min_distance
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
        
        return detected_faces
        
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
        return []

@app.route('/')
def index():
    """Main page with mobile optimized interface"""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>Face Recognition System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
                padding: 10px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            
            h1 {
                text-align: center;
                margin-bottom: 20px;
                color: white;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                font-size: 1.8em;
            }
            
            .section {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
            
            .camera-container {
                position: relative;
                width: 100%;
                border-radius: 12px;
                overflow: hidden;
                background: #f0f0f0;
                aspect-ratio: 4/3;
                margin-bottom: 15px;
            }
            
            #videoElement {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            
            .camera-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(0,0,0,0.8);
                color: white;
                text-align: center;
                padding: 20px;
            }
            
            .controls {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            button {
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                cursor: pointer;
                flex: 1;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            button.stop {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            }
            
            .results {
                margin-top: 15px;
            }
            
            .face-result {
                padding: 12px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .face-known {
                color: #27ae60;
                font-weight: 600;
            }
            
            .face-unknown {
                color: #e74c3c;
                font-weight: 600;
            }
            
            .upload-section h2 {
                margin-bottom: 15px;
                color: #333;
            }
            
            .upload-form {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            input[type="text"] {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }
            
            button.upload {
                background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                color: white;
            }
            
            .status {
                padding: 12px;
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                border-radius: 8px;
                margin-bottom: 15px;
                text-align: center;
                font-weight: 500;
            }
            
            #fileInput {
                display: none;
            }
            
            .file-upload-btn {
                display: inline-block;
                padding: 12px 20px;
                background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
                color: white;
                border-radius: 8px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            
            .file-upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            
            .file-name {
                margin-top: 8px;
                font-size: 14px;
                color: #666;
                text-align: center;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white;
                padding: 12px;
                border-radius: 8px;
                margin: 10px 0;
            }
            
            .success {
                background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                color: white;
                padding: 12px;
                border-radius: 8px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Face Recognition System</h1>
            
            <div class="section">
                <div class="camera-container">
                    <video id="videoElement" autoplay muted playsinline></video>
                    <div class="camera-overlay" id="cameraOverlay">
                        <div>
                            <div style="font-size: 48px; margin-bottom: 12px;">📸</div>
                            <div>Click "Start" to enable camera</div>
                        </div>
                    </div>
                </div>
                
                <div class="controls">
                    <button id="startBtn">🚀 Start</button>
                    <button id="stopBtn" class="stop" disabled>⏹️ Stop</button>
                </div>
                
                <div class="status" id="status">
                    System ready - Click "Start" to begin
                </div>
                
                <div class="results" id="results">
                    <p style="text-align: center; color: #666;">Recognition results will appear here...</p>
                </div>
            </div>
            
            <div class="section upload-section">
                <h2>➕ Add New Face</h2>
                <form class="upload-form" id="uploadForm" enctype="multipart/form-data">
                    <div>
                        <label for="nameInput">Person's Name:</label>
                        <input type="text" id="nameInput" required placeholder="Enter name" maxlength="50">
                    </div>
                    
                    <div>
                        <label for="fileInput" class="file-upload-btn">
                            📁 Select Image
                            <input type="file" id="fileInput" accept="image/*" required>
                        </label>
                        <div class="file-name" id="fileName">No file selected</div>
                    </div>
                    
                    <button type="submit" class="upload">🎯 Upload and Train</button>
                </form>
                <div id="uploadStatus"></div>
            </div>
        </div>
        
        <canvas id="canvas" style="display: none;"></canvas>
        
        <script>
            const video = document.getElementById('videoElement');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const results = document.getElementById('results');
            const cameraOverlay = document.getElementById('cameraOverlay');
            const uploadForm = document.getElementById('uploadForm');
            const nameInput = document.getElementById('nameInput');
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const uploadStatus = document.getElementById('uploadStatus');
            
            let stream = null;
            let isRecognizing = false;
            let recognitionInterval = null;
            let lastAnalysisTime = 0;
            
            // File input change handler
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    const file = e.target.files[0];
                    if (file.size > 16 * 1024 * 1024) {
                        alert('File size too large. Please select a file smaller than 16MB.');
                        e.target.value = '';
                        fileName.textContent = 'No file selected';
                        return;
                    }
                    fileName.textContent = file.name;
                } else {
                    fileName.textContent = 'No file selected';
                }
            });
            
            // Upload form handler
            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const name = nameInput.value.trim();
                if (!name) {
                    uploadStatus.innerHTML = '<div class="error">❌ Please enter a name</div>';
                    return;
                }
                
                if (!fileInput.files[0]) {
                    uploadStatus.innerHTML = '<div class="error">❌ Please select an image</div>';
                    return;
                }
                
                const formData = new FormData();
                formData.append('name', name);
                formData.append('file', fileInput.files[0]);
                
                try {
                    uploadStatus.innerHTML = '<div class="status"><div class="loading"></div>Uploading and training...</div>';
                    
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error('Server returned non-JSON response');
                    }
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        uploadStatus.innerHTML = `<div class="success">✅ ${data.message}</div>`;
                        nameInput.value = '';
                        fileInput.value = '';
                        fileName.textContent = 'No file selected';
                        
                        // Reload embeddings
                        setTimeout(() => {
                            checkSystemStatus();
                        }, 1000);
                    } else {
                        uploadStatus.innerHTML = `<div class="error">❌ Error: ${data.message || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    uploadStatus.innerHTML = `<div class="error">❌ Network error: ${error.message}</div>`;
                }
            });
            
            // Check system status
            async function checkSystemStatus() {
                try {
                    const response = await fetch('/health');
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    if (!data.embeddings_loaded) {
                        status.innerHTML = '⚠️ No registered faces. Please add images.';
                        startBtn.disabled = true;
                        return false;
                    }
                    
                    status.innerHTML = `✅ System ready - ${data.registered_faces.length} face(s) registered`;
                    startBtn.disabled = false;
                    return true;
                } catch (error) {
                    console.error('Error checking system status:', error);
                    status.innerHTML = '❌ Unable to connect to server';
                    return false;
                }
            }
            
            // Start camera
            async function startCamera() {
                try {
                    const constraints = {
                        video: { 
                            facingMode: 'user',
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        }
                    };
                    
                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = stream;
                    
                    video.addEventListener('loadedmetadata', () => {
                        cameraOverlay.style.display = 'none';
                    });
                    
                    status.innerHTML = '✅ Camera active - Analyzing...';
                    return true;
                } catch (err) {
                    console.error('Camera error:', err);
                    let errorMsg = '❌ Camera error: ';
                    
                    if (err.name === 'NotAllowedError') {
                        errorMsg += 'Permission denied. Please allow camera access.';
                    } else if (err.name === 'NotFoundError') {
                        errorMsg += 'No camera found.';
                    } else {
                        errorMsg += 'Could not access camera.';
                    }
                    
                    status.innerHTML = errorMsg;
                    return false;
                }
            }
            
            // Stop camera
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                video.srcObject = null;
                cameraOverlay.style.display = 'flex';
                status.innerHTML = '⏹️ Camera stopped';
            }
            
            // Capture and analyze image
            function captureAndAnalyze() {
                const now = Date.now();
                
                if (now - lastAnalysisTime < 2000) return;
                lastAnalysisTime = now;
                
                if (!video.videoWidth || !video.videoHeight) return;
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Analysis error:', error);
                    status.innerHTML = '❌ Analysis error - Check connection';
                });
            }
            
            // Display results
            function displayResults(data) {
                if (data.success && data.faces && data.faces.length > 0) {
                    results.innerHTML = data.faces.map(face => {
                        const confidence = Math.round(face.confidence * 100);
                        const isKnown = face.name !== 'Unknown';
                        
                        return `
                            <div class="face-result ${isKnown ? 'face-known' : 'face-unknown'}">
                                <span>${isKnown ? '👤' : '❓'} ${face.name}</span>
                                <span>${confidence}%</span>
                            </div>
                        `;
                    }).join('');
                    
                    const knownFaces = data.faces.filter(f => f.name !== 'Unknown').length;
                    status.innerHTML = `🔍 ${data.faces.length} face(s) detected (${knownFaces} recognized)`;
                } else {
                    status.innerHTML = '👁️ Looking for faces...';
                    results.innerHTML = '<p style="text-align: center; color: #666;">No faces detected</p>';
                }
            }
            
            // Button events
            startBtn.addEventListener('click', async () => {
                if (!(await checkSystemStatus())) return;
                
                if (await startCamera()) {
                    isRecognizing = true;
                    recognitionInterval = setInterval(captureAndAnalyze, 3000);
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                }
            });
            
            stopBtn.addEventListener('click', () => {
                isRecognizing = false;
                if (recognitionInterval) {
                    clearInterval(recognitionInterval);
                    recognitionInterval = null;
                }
                stopCamera();
                startBtn.disabled = false;
                stopBtn.disabled = true;
                results.innerHTML = '<p style="text-align: center; color: #666;">Results will appear here...</p>';
            });
            
            // Initialize
            checkSystemStatus();
        </script>
    </body>
    </html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze image received from client"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'faces': []
            }), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data',
                'faces': []
            }), 400
        
        image_data = data['image']
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid base64 image data',
                'faces': []
            }), 400
        
        # Convert to numpy array
        pil_image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process image
        detected_faces = process_frame(frame)
        
        return jsonify({
            'success': True,
            'faces': detected_faces,
            'count': len(detected_faces)
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'faces': []
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and add to database"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        if 'name' not in request.form:
            return jsonify({'success': False, 'message': 'No name provided'}), 400
        
        file = request.files['file']
        name = request.form['name'].strip()
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not name:
            return jsonify({'success': False, 'message': 'Name cannot be empty'}), 400
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'message': 'Invalid file type. Use JPG, PNG, or BMP.'}), 400
        
        # Create data directory if not exists
        os.makedirs(INPUT_DIR, exist_ok=True)
        
        # Save the original file with timestamp to avoid conflicts
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.{file_ext}"
        filepath = os.path.join(INPUT_DIR, filename)
        
        try:
            file.save(filepath)
            logger.info(f"Saved uploaded file: {filepath}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return jsonify({'success': False, 'message': 'Failed to save file'}), 500
        
        # Process the image and update embeddings
        try:
            if setup_face_recognition():
                if load_embeddings():
                    return jsonify({
                        'success': True,
                        'message': f'Successfully added {name} to the database'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to load updated embeddings'
                    }), 500
            else:
                # Clean up the uploaded file if processing failed
                try:
                    os.remove(filepath)
                except:
                    pass
                return jsonify({
                    'success': False,
                    'message': 'Failed to process the image. Please ensure the image contains a clear face.'
                }), 500
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({
                'success': False,
                'message': f'Processing error: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({
            'success': False,
            'message': f'Upload error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'embeddings_loaded': len(embs) > 0,
            'registered_faces': list(embs.keys()) if embs else [],
            'model': MODEL_NAME,
            'setup_complete': setup_complete,
            'timestamp': int(time.time())
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/setup', methods=['POST'])
def manual_setup():
    """Manual setup trigger endpoint"""
    try:
        success = setup_face_recognition()
        if success:
            load_embeddings()
        return jsonify({
            'success': success,
            'message': 'Setup completed successfully' if success else 'Setup failed - check if images are in data directory'
        })
    except Exception as e:
        logger.error(f"Manual setup error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

def main():
    """Main function to start the application"""
    try:
        logger.info("🚀 Starting Face Recognition System...")
        logger.info("=" * 60)
        
        # Create necessary directories
        for directory in [INPUT_DIR, CROPPED_DIR, EMBEDDINGS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Try to load existing embeddings, setup if needed
        if not load_embeddings():
            logger.warning("⚠️ No embeddings found, running automatic setup...")
            if setup_face_recognition():
                load_embeddings()
            else:
                logger.warning("❌ Automatic setup failed")
                logger.info("📁 Please add images to the './data' directory")
                logger.info("🔄 Or use the web interface to upload images")
        
        logger.info("=" * 60)
        logger.info("✅ System ready!")
        logger.info("📱 Access the application via your web browser")
        
        # Get port from environment (for deployment)
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        
        # Run the app
        app.run(host=host, port=port, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == '__main__':
    main()
