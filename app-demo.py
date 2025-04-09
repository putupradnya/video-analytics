import cv2
import torch
from ultralytics import YOLO
from flask import Flask, Response, render_template_string
import threading
import time

# Load model
model = YOLO("yolov8n.pt")
device = 0 if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Using CUDA:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")



# Video path
video_path = "sample/test-1.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Shared frame
lock = threading.Lock()
annotated_frame = None

def inference_loop():
    global annotated_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_resized = cv2.resize(frame, (480, 270))

        results = model.predict(frame_resized, device=device, verbose=False)
        annotated = results[0].plot()

        with lock:
            annotated_frame = annotated

        time.sleep(0.03)  # ~30 FPS

# Start inference thread
threading.Thread(target=inference_loop, daemon=True).start()

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
    <html>
        <head><title>YOLO Video Stream</title></head>
        <body>
            <h1>Realtime YOLO Detection</h1>
            <img src="{{ url_for('video_feed') }}">
        </body>
    </html>
    ''')

def generate_frames():
    global annotated_frame
    while True:
        with lock:
            if annotated_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, threaded=True)
