"""
Vehicle Number Plate Detection - With Plate Localization
Uses plate detection module for accurate number plate recognition
"""
import os
import cv2
import threading
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from plate_detector import detect_and_read_plate, detect_from_camera_frame, detect_and_read_plate_fast, detect_mobile_plate, detect_instant

app = Flask(__name__)
CORS(app)

# Global state
detection_result = {"text": "No plate detected", "confidence": 0}
ocr_lock = False
camera = None
# Add stability - timestamp of last manual detection
last_manual_detection = 0
MANUAL_DETECTION_COOLDOWN = 5  # seconds

def ocr_worker(frame):
    """Background OCR with plate detection"""
    global detection_result, ocr_lock, last_manual_detection
    import time
    try:
        # Skip if within cooldown period from manual detection
        if time.time() - last_manual_detection < MANUAL_DETECTION_COOLDOWN:
            print("Skipping - within cooldown period", flush=True)
            return
        
        # Use the plate detector module
        result = detect_from_camera_frame(frame)
        
        if result["text"] and result["text"] != "No plate detected":
            print(f"DETECTED: {result['text']} ({result['confidence']})", flush=True)
            detection_result = result
        else:
            print("No plate detected in this frame", flush=True)
    except Exception as e:
        print(f"Detection Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        ocr_lock = False

def open_camera():
    """Open camera with retries"""
    # Check if we're in a cloud environment (no camera available)
    import os
    RENDER_ENV = os.environ.get('RENDER', False) or os.environ.get('PORT', False)
    
    # Skip camera in cloud environments
    if RENDER_ENV:
        print("Running in cloud environment - using sample image mode", flush=True)
        return None
    
    global camera
    for attempt in range(3):
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            # Higher resolution for better detection
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 30)
            print(f"Camera opened on attempt {attempt+1}", flush=True)
            return cam
        print(f"Camera attempt {attempt+1} failed", flush=True)
        import time
        time.sleep(0.5)
    return None

def generate_frames():
    """Video streaming with detection overlay"""
    global ocr_lock, camera, detection_result
    
    # Open camera once for this generator
    if camera is None or not camera.isOpened():
        camera = open_camera()
    
    if camera is None or not camera.isOpened():
        # Fallback to sample image
        sample_path = os.path.join(os.path.dirname(__file__), "81suTTBxfuL.jpg")
        if os.path.exists(sample_path):
            frame = cv2.imread(sample_path)
            while True:
                # Draw detection result on frame if available
                if detection_result and detection_result.get("text") and detection_result.get("text") != "No plate detected":
                    cv2.putText(frame, f"PLATE: {detection_result['text']}", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"CONF: {int(detection_result['confidence']*100)}%", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    print("Streaming started", flush=True)
    frame_count = 0
    
    while True:
        if not camera.isOpened():
            # Try to reopen
            camera = open_camera()
            if camera is None:
                break
        
        success, frame = camera.read()
        if not success:
            continue
        
        frame_count += 1
        
        # Run OCR on EVERY frame for instant detection (no skipping)
        if not ocr_lock:
            ocr_lock = True
            thread = threading.Thread(target=ocr_worker, args=(frame.copy(),))
            thread.daemon = True
            thread.start()
        
        # Draw detection result on the frame
        if detection_result and detection_result.get("text") and detection_result.get("text") != "No plate detected":
            # Draw a rectangle and text on the frame
            cv2.putText(frame, f"PLATE: {detection_result['text']}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"CONF: {int(detection_result['confidence']*100)}%", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw green rectangle
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 3)
        
        # Encode and stream with higher quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    return jsonify(detection_result)

@app.route('/status')
def status():
    return jsonify({"current_plate": detection_result["text"], "detection_active": True})

@app.route('/upload', methods=['POST'])
def upload_and_detect():
    """Upload image directly for INSTANT detection - optimized for mobile images"""
    if 'image' not in request.files:
        return jsonify({"text": "No image provided", "confidence": 0})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"text": "No file selected", "confidence": 0})
    
    # Read image from upload
    import numpy as np
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"text": "Invalid image", "confidence": 0})
    
    # Use INSTANT detection optimized for mobile images
    text, conf = detect_instant(image)
    return jsonify({"text": text, "confidence": conf})


@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Detect plate from uploaded image - uses instant detection"""
    if 'image' not in request.files:
        sample_path = os.path.join(os.path.dirname(__file__), "81suTTBxfuL.jpg")
        if os.path.exists(sample_path):
            image = cv2.imread(sample_path)
            text, conf = detect_instant(image)
            return jsonify({"text": text, "confidence": conf})
        return jsonify({"text": "No image provided", "confidence": 0})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"text": "No file selected", "confidence": 0})
    
    filename = f"upload_{int(cv2.getTickCount())}.jpg"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file.save(filepath)
    
    try:
        image = cv2.imread(filepath)
        text, conf = detect_instant(image)
        return jsonify({"text": text, "confidence": conf})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/detect_sample', methods=['GET'])
def detect_sample():
    """Detect from sample image using instant detection"""
    global detection_result, last_manual_detection
    import time
    sample_path = os.path.join(os.path.dirname(__file__), "81suTTBxfuL.jpg")
    if os.path.exists(sample_path):
        image = cv2.imread(sample_path)
        text, conf = detect_instant(image)
        # Update global result and set cooldown
        detection_result = {"text": text, "confidence": conf}
        last_manual_detection = time.time()
        return jsonify({"text": text, "confidence": conf})
    return jsonify({"text": "Sample image not found", "confidence": 0})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    return jsonify({"detection_active": True})

@app.route('/toggle', methods=['POST'])
def toggle():
    return jsonify({"active": True})

if __name__ == '__main__':
    print("Starting app at http://localhost:5000", flush=True)
    print("Using Plate Detection Module with localization", flush=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

