import cv2
import numpy as np
import torch
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import threading
import time
import os
import requests
from dotenv import load_dotenv
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Load .env
load_dotenv()

app = Flask(__name__)

# Config
VIDEO_OPTIONS = {
    "Video 1 (RPTRA A)": "sample/test-1.mp4",
    "Video 2 (RPTRA B)": "sample/test-22.mp4",
    "Webcam": 0
}
MODEL_OPTIONS = {
    "YOLOv8n": "yolov8n.pt",
    "MobileNet SSD": "mobilenet_ssd"
}

# Telegram Config
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message, photo_path=None):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    requests.get(url, params=params)

    if photo_path:
        url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": CHAT_ID}
            requests.post(url_photo, data=data, files=files)

# Shared Variables
frame_lock = threading.Lock()
latest_frame = None
selected_video = list(VIDEO_OPTIONS.values())[0]
selected_model = list(MODEL_OPTIONS.values())[0]
line_orientation = "Horizontal"
line_position = 200
up_count = 0
down_count = 0
alert_50_sent = False
alert_75_sent = False
inference_started = False
running = False
video_thread = None

# Load Models
model_yolo = YOLO("yolov8n.pt")
net_ssd = cv2.dnn.readNetFromCaffe("sample/deploy.prototxt", "sample/mobilenet_iter_73000.caffemodel")
net_ssd.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_ssd.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

tracked_objects = {}
next_id = 1
frame_count = 0
frame_limit = 10
distance_threshold = 80

def process_video():
    global latest_frame, up_count, down_count, tracked_objects, next_id
    global alert_50_sent, alert_75_sent, frame_count, running

    cap = cv2.VideoCapture(selected_video)
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        (h, w) = frame.shape[:2]
        detected_centroids = []

        if selected_model == "YOLOv8n":
            results = model_yolo(frame, batch=4, iou=0.7)
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result.tolist()
                if int(cls) == 0 and conf > 0.5:
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    detected_centroids.append(centroid)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        else:
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net_ssd.setInput(blob)
            detections = net_ssd.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 15:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        detected_centroids.append(centroid)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        existing_ids = list(tracked_objects.keys())
        prev_centroids = [tracked_objects[obj_id][0] for obj_id in existing_ids]

        cost_matrix = np.zeros((len(prev_centroids), len(detected_centroids)))
        for i, prev_c in enumerate(prev_centroids):
            for j, new_c in enumerate(detected_centroids):
                cost_matrix[i, j] = distance.euclidean(prev_c, new_c)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        new_tracked_objects = {}

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < distance_threshold:
                obj_id = existing_ids[i]
                prev_centroid, last_seen, entry_frame = tracked_objects[obj_id]

                if line_orientation == "Horizontal":
                    if prev_centroid[1] < line_position and detected_centroids[j][1] > line_position:
                        down_count += 1
                    elif prev_centroid[1] > line_position and detected_centroids[j][1] < line_position:
                        up_count += 1
                else:
                    if prev_centroid[0] < line_position and detected_centroids[j][0] > line_position:
                        down_count += 1
                    elif prev_centroid[0] > line_position and detected_centroids[j][0] < line_position:
                        up_count += 1

                new_tracked_objects[obj_id] = (detected_centroids[j], frame_count, entry_frame)

        for j, new_c in enumerate(detected_centroids):
            if j not in col_ind:
                new_tracked_objects[next_id] = (new_c, frame_count, frame_count)
                next_id += 1

        tracked_objects = new_tracked_objects

        total_visitor = down_count - up_count
        font = cv2.FONT_HERSHEY_SIMPLEX

        if line_orientation == "Horizontal":
            cv2.line(frame, (0, line_position), (w, line_position), (0, 255, 255), 2)
            cv2.putText(frame, "Masuk", (10, line_position + 25), font, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Keluar", (10, line_position - 10), font, 0.5, (0, 0, 255), 1)
        else:
            cv2.line(frame, (line_position, 0), (line_position, h), (0, 255, 255), 2)
            cv2.putText(frame, "Masuk", (line_position + 10, 30), font, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Keluar", (line_position - 70, 30), font, 0.5, (0, 0, 255), 1)

        if total_visitor >= 5 and not alert_50_sent:
            message = "⚠️ Alert! Pengunjung mencapai 50% kapasitas"
            screenshot_path = "screenshot_50.jpg"
            cv2.imwrite(screenshot_path, frame)
            send_telegram_message(message, screenshot_path)
            alert_50_sent = True

        elif total_visitor >= 7 and not alert_75_sent:
            message = "🚨 Alert! Pengunjung mencapai 75% kapasitas"
            screenshot_path = "screenshot_75.jpg"
            cv2.imwrite(screenshot_path, frame)
            send_telegram_message(message, screenshot_path)
            alert_75_sent = True

        with frame_lock:
            latest_frame = cv2.imencode('.jpg', frame)[1].tobytes()

        time.sleep(0.03)

    cap.release()
    running = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_video, selected_model, line_orientation, line_position
    global up_count, down_count, alert_50_sent, alert_75_sent, running, video_thread, inference_started

    if request.method == 'POST':
        selected_video = VIDEO_OPTIONS.get(request.form['video_source'], selected_video)
        selected_model = MODEL_OPTIONS.get(request.form['model_choice'], selected_model)
        line_orientation = request.form['line_orientation']
        line_position = int(request.form['line_position'])

        up_count = 0
        down_count = 0
        alert_50_sent = False
        alert_75_sent = False

        if inference_started:
            running = False
            time.sleep(1)
            video_thread = threading.Thread(target=process_video, daemon=True)
            video_thread.start()

        return redirect(url_for('index'))

    total_visitor = down_count - up_count
    return render_template("dashboard.html",
        video_options=VIDEO_OPTIONS,
        model_options=MODEL_OPTIONS,
        current_video=[k for k, v in VIDEO_OPTIONS.items() if v == selected_video][0],
        current_model=[k for k, v in MODEL_OPTIONS.items() if v == selected_model][0],
        line_orientation=line_orientation,
        line_position=line_position,
        up_count=up_count,
        down_count=down_count,
        total_visitor=total_visitor,
        alert_message=request.args.get("alert_message")
    )

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    total_visitor = down_count - up_count
    return jsonify({"up": up_count, "down": down_count, "total": total_visitor})

@app.route('/reset', methods=['POST'])
def reset():
    global up_count, down_count, alert_50_sent, alert_75_sent
    up_count = 0
    down_count = 0
    alert_50_sent = False
    alert_75_sent = False
    return redirect(url_for('index', alert_message="Counter berhasil direset."))

@app.route('/start', methods=['POST'])
def start():
    global inference_started, video_thread
    if not inference_started:
        video_thread = threading.Thread(target=process_video, daemon=True)
        video_thread.start()
        inference_started = True
    return redirect(url_for('index', alert_message="Inference dimulai."))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
