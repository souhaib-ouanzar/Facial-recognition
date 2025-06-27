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

app = Flask(__name__)

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

def crop_faces(input_dir, output_dir, detector_backend="yolov8"):
    """Crop faces from images using your optimized function"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Cropping faces from '{input_dir}' to '{output_dir}'...")
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"❌ No image files found in '{input_dir}'")
        return False
    
    success_count = 0
    for img_file in tqdm(image_files, desc="Cropping faces"):
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
            print(f"❌ Error processing {img_file}: {str(e)}")
    
    print(f"✅ Successfully cropped {success_count}/{len(image_files)} faces")
    return success_count > 0

def extract_embeddings(cropped_faces_dir, output_dir, model_name="Facenet512"):
    """Extract face embeddings from cropped face images"""
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings = {}
    
    if not os.path.exists(cropped_faces_dir):
        print(f"❌ Cropped faces directory '{cropped_faces_dir}' not found!")
        return False
    
    image_files = [f for f in os.listdir(cropped_faces_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"❌ No image files found in '{cropped_faces_dir}'")
        return False
    
    print(f"🔄 Extracting embeddings from {len(image_files)} images...")
    
    success_count = 0
    for img_file in tqdm(image_files, desc="Extracting embeddings"):
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
            print(f"❌ Error processing {img_file}: {str(e)}")
    
    if embeddings:
        output_path = os.path.join(output_dir, f"embs_{model_name.lower()}.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
        
        print(f"✅ Successfully saved {len(embeddings)} embeddings to: {output_path}")
        print(f"🏷️  Registered faces: {list(embeddings.keys())}")
        return True
    else:
        print("❌ No embeddings were extracted!")
        return False

def setup_face_recognition():
    """Complete setup process for face recognition"""
    global setup_complete
    
    print("🚀 Starting Face Recognition Setup...")
    
    # Check if data directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        print("📁 Please create the directory and add images of people you want to recognize.")
        return False
    
    # Check if there are images in data directory
    image_files = [f for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"❌ No images found in '{INPUT_DIR}' directory!")
        print("📸 Please add images of people you want to recognize.")
        return False
    
    print(f"📸 Found {len(image_files)} images in data directory")
    
    # Step 1: Crop faces
    if not crop_faces(INPUT_DIR, CROPPED_DIR):
        return False
    
    # Step 2: Extract embeddings
    if not extract_embeddings(CROPPED_DIR, EMBEDDINGS_DIR, MODEL_NAME):
        return False
    
    print("✅ Face Recognition Setup Complete!")
    setup_complete = True
    return True

def load_embeddings():
    """Load face embeddings from file"""
    global embs
    try:
        with open(EMBEDDINGS_PATH, "rb") as file:
            embs = pickle.load(file)
            print(f"✅ Loaded {len(embs)} face embeddings")
            return True
    except FileNotFoundError:
        print(f"❌ Embeddings file not found: {EMBEDDINGS_PATH}")
        print("🔄 Running setup process...")
        if setup_face_recognition():
            return load_embeddings()  # Retry loading after setup
        return False
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False

def process_frame(frame):
    """Process frame and detect faces"""
    try:
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
                    print(f"❌ Error processing face: {e}")
                    continue
        
        return detected_faces
        
    except Exception as e:
        print(f"❌ Error in frame processing: {e}")
        return []

@app.route('/')
def index():
    """Page principale avec interface mobile optimisée"""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>Face Recognition</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }
            
            body {
                font-family: Arial, sans-serif;
                background: white;
                color: #333;
                padding: 20px;
                line-height: 1.6;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            
            h1 {
                text-align: center;
                margin-bottom: 20px;
                color: #444;
            }
            
            .section {
                background: #f9f9f9;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .camera-container {
                position: relative;
                width: 100%;
                border-radius: 8px;
                overflow: hidden;
                background: #eee;
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
                background: rgba(0,0,0,0.7);
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
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                background: #4285f4;
                color: white;
                cursor: pointer;
                flex: 1;
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            button.stop {
                background: #ea4335;
            }
            
            .results {
                margin-top: 15px;
            }
            
            .face-result {
                padding: 10px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
            }
            
            .face-known {
                color: #34a853;
            }
            
            .face-unknown {
                color: #ea4335;
            }
            
            .upload-section {
                margin-top: 30px;
            }
            
            .upload-form {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            input, button.upload {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            
            button.upload {
                background: #34a853;
                color: white;
            }
            
            .status {
                padding: 10px;
                background: #e8f0fe;
                border-radius: 4px;
                margin-bottom: 15px;
                text-align: center;
            }
            
            #fileInput {
                display: none;
            }
            
            .file-upload-btn {
                display: inline-block;
                padding: 10px 15px;
                background: #f1f1f1;
                border: 1px dashed #ccc;
                border-radius: 4px;
                text-align: center;
                cursor: pointer;
            }
            
            .file-name {
                margin-top: 5px;
                font-size: 14px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Face Recognition System</h1>
            
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
                    <button id="startBtn">Start</button>
                    <button id="stopBtn" class="stop" disabled>Stop</button>
                </div>
                
                <div class="status" id="status">
                    System ready - Click "Start" to begin
                </div>
                
                <div class="results" id="results">
                    <p>Recognition results will appear here...</p>
                </div>
            </div>
            
            <div class="section upload-section">
                <h2>Add New Face</h2>
                <form class="upload-form" id="uploadForm" enctype="multipart/form-data">
                    <div>
                        <label for="nameInput">Person's Name:</label>
                        <input type="text" id="nameInput" required placeholder="Enter name">
                    </div>
                    
                    <div>
                        <label for="fileInput" class="file-upload-btn">
                            Select Image
                            <input type="file" id="fileInput" accept="image/*" required>
                        </label>
                        <div class="file-name" id="fileName">No file selected</div>
                    </div>
                    
                    <button type="submit" class="upload">Upload and Train</button>
                </form>
                <div id="uploadStatus"></div>
            </div>
        </div>
        
        <canvas id="canvas"></canvas>
        
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
                    fileName.textContent = e.target.files[0].name;
                } else {
                    fileName.textContent = 'No file selected';
                }
            });
            
            // Upload form handler
            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('name', nameInput.value);
                formData.append('file', fileInput.files[0]);
                
                try {
                    uploadStatus.innerHTML = '<p>Uploading and training...</p>';
                    
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        uploadStatus.innerHTML = `<p style="color:green;">✅ ${data.message}</p>`;
                        nameInput.value = '';
                        fileInput.value = '';
                        fileName.textContent = 'No file selected';
                        
                        // Reload embeddings
                        await checkSystemStatus();
                    } else {
                        uploadStatus.innerHTML = `<p style="color:red;">❌ Error: ${data.message || 'Unknown error'}</p>`;
                    }
                } catch (error) {
                    uploadStatus.innerHTML = `<p style="color:red;">❌ Network error: ${error.message}</p>`;
                }
            });
            
            // Check system status
            async function checkSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    if (!data.embeddings_loaded) {
                        status.innerHTML = '⚠️ No registered faces. Please add images.';
                        startBtn.disabled = true;
                        return false;
                    }
                    return true;
                } catch (error) {
                    console.error('Error checking system status:', error);
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
                
                if (now - lastAnalysisTime < 1500) return;
                lastAnalysisTime = now;
                
                if (!video.videoWidth || !video.videoHeight) return;
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.7);
                
                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                })
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Analysis error:', error);
                    status.innerHTML = '❌ Analysis error - Reconnecting...';
                });
            }
            
            // Display results
            function displayResults(data) {
                if (data.faces && data.faces.length > 0) {
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
                    
                    let statusText = `🔍 ${data.faces.length} face(s) detected`;
                    status.innerHTML = statusText;
                } else {
                    status.innerHTML = '👁️ Looking for faces...';
                    results.innerHTML = '<p>No faces detected</p>';
                }
            }
            
            // Button events
            startBtn.addEventListener('click', async () => {
                if (!(await checkSystemStatus())) return;
                
                if (await startCamera()) {
                    isRecognizing = true;
                    recognitionInterval = setInterval(captureAndAnalyze, 2000);
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
                results.innerHTML = '<p>Results will appear here...</p>';
            });
            
            // Initialize
            checkSystemStatus();
        </script>
    </body>
    </html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze image received from phone"""
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
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
        print(f"❌ Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'faces': []
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and add to database"""
    try:
        if 'file' not in request.files or 'name' not in request.form:
            return jsonify({'success': False, 'message': 'Missing file or name'})
        
        file = request.files['file']
        name = request.form['name']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        
        if not name:
            return jsonify({'success': False, 'message': 'No name provided'})
        
        # Create data directory if not exists
        os.makedirs(INPUT_DIR, exist_ok=True)
        
        # Save the original file
        filename = f"{name}_{int(time.time())}.{file.filename.split('.')[-1]}"
        filepath = os.path.join(INPUT_DIR, filename)
        file.save(filepath)
        
        # Process the image and update embeddings
        if setup_face_recognition():
            load_embeddings()
            return jsonify({
                'success': True,
                'message': f'Successfully added {name} to the database'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to process the image'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'embeddings_loaded': len(embs) > 0,
        'registered_faces': list(embs.keys()) if embs else [],
        'model': MODEL_NAME,
        'setup_complete': setup_complete
    })

@app.route('/setup', methods=['POST'])
def manual_setup():
    """Manual setup trigger endpoint"""
    try:
        success = setup_face_recognition()
        if success:
            load_embeddings()
        return jsonify({
            'success': success,
            'message': 'Setup completed successfully' if success else 'Setup failed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def main():
    print("🚀 Starting Face Recognition System...")
    print("=" * 60)
    
    # Try to load existing embeddings, setup if needed
    if not load_embeddings():
        print("⚠️ No embeddings found, running automatic setup...")
        if not setup_face_recognition():
            print("❌ Automatic setup failed")
            print("📁 Please add images to the './data' directory")
            print("🔄 Then restart the application")
        else:
            load_embeddings()
    
    print("=" * 60)
    print("✅ System ready!")
    print("📱 Access the application via your mobile browser")
    print("🌐 Local URL: http://localhost:5000")
    
    # Get port from environment (for deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    main()