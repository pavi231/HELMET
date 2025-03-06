from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import os

app = Flask(__name__)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detection_running = False
helmet_detected = False
lock = threading.Lock()

def generate_frames():
    global detection_running, helmet_detected
    cap = cv2.VideoCapture(0)  # Change index if needed (0 or 1)
    
    while detection_running:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        helmet_detected = len(faces) == 0  # If face is detected, helmet is missing

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "No Helmet Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running
    with lock:
        if not detection_running:
            detection_running = True
            threading.Thread(target=generate_frames, daemon=True).start()
    return jsonify(status='started')

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running
    with lock:
        detection_running = False
    return jsonify(status='stopped')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(helmet_detected=helmet_detected)

if __name__ == '__main__':
    app.run(debug=True)