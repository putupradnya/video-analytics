import streamlit as st
import cv2
import numpy as np
import time
import torch
import boto3
from ultralytics import YOLO
import requests
import os
from io import BytesIO
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# **🔹 Konfigurasi AWS S3**
S3_BUCKET_NAME = "demo-jsc"
VIDEO_PATH_PREFIX = "rptra/sample/"
MODEL_PATH_PREFIX = "rptra/models/"

s3_client = boto3.client("s3")

def load_from_s3(s3_key):
    """Mengambil file dari S3 dan menyimpannya sebagai file sementara."""
    local_filename = f"/tmp/{os.path.basename(s3_key)}"
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_filename)
        return local_filename
    except Exception as e:
        st.error(f"Error mengambil file dari S3: {s3_key} - {e}")
        return None

# **🔹 Konfigurasi Streamlit UI**
st.set_page_config(page_title="People Counting CCTV", layout="wide")
st.title("RPTRA CCTV Monitoring System")
st.write("Tracking real-time movement using object detection.")

# **🔹 Konfigurasi Telegram**
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

# **🔹 Pilihan Video & Model**
video_options = {
    "Video 1 (RPTRA A)": f"{VIDEO_PATH_PREFIX}test-1.mp4",
    "Video 2 (RPTRA B)": f"{VIDEO_PATH_PREFIX}test-22.mp4",
}

model_options = {
    "YOLOv8n": f"{MODEL_PATH_PREFIX}yolov8n.pt",
    "MobileNet SSD": f"{MODEL_PATH_PREFIX}mobilenet_iter_73000.caffemodel",
}

col1, col2, col3 = st.columns(3)

with col1:
    selected_video = st.selectbox("Sumber Video:", list(video_options.keys()), index=0)

with col2:
    selected_model = st.selectbox("Model Deteksi:", list(model_options.keys()), index=0)

with col3:
    line_orientation = st.radio("🔄 Orientasi Garis Deteksi:", ["Horizontal", "Vertikal"])
    line_position = st.slider("⚙️ Posisi Garis Deteksi (pixel)", 100, 500, 200)

# **🔹 Load Model dari S3**
model_path = load_from_s3(model_options[selected_model])
if model_path:
    if selected_model == "YOLOv8n":
        model = YOLO(model_path)
        model.to("cpu")  
    else:
        net = cv2.dnn.readNetFromCaffe("sample/deploy.prototxt", model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
else:
    st.error("Gagal memuat model, pastikan file tersedia di S3.")

# **🔹 UI Metrics**
col1, col2, col3 = st.columns(3)
total_people_placeholder = col1.empty()
up_count_placeholder = col2.empty()
down_count_placeholder = col3.empty()

# **🔹 Load Video dari S3**
video_path = load_from_s3(video_options[selected_video])
cap = cv2.VideoCapture(video_path) if video_path else None

video_placeholder = st.empty()
screenshot_placeholder = st.empty()
alert_50_sent = False
alert_75_sent = False

# **🔹 Variabel Tracking**
tracked_objects = {}
next_id = 1
up_count = 0
down_count = 0
frame_count = 0
distance_threshold = 80

# **🔹 Loop Video**
if cap:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        (h, w) = frame.shape[:2]
        detected_centroids = []

        # **🔹 Tambahkan Label "Masuk" & "Keluar" di Garis**
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1

        # **🔹 Garis Deteksi**
        if line_orientation == "Horizontal":
            cv2.line(frame, (0, line_position), (w, line_position), (0, 255, 255), 2)
            cv2.putText(frame, "Masuk", (10, line_position + 25), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
            cv2.putText(frame, "Keluar", (10, line_position - 10), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        else:
            cv2.line(frame, (line_position, 0), (line_position, h), (0, 255, 255), 2)
            cv2.putText(frame, "Masuk", (line_position + 10, 30), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
            cv2.putText(frame, "Keluar", (line_position - 70, 30), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        # **🔹 Deteksi YOLOv8**
        if selected_model == "YOLOv8n":
            results = model(frame)
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result.tolist()
                if int(cls) == 0 and conf > 0.5:
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    detected_centroids.append(centroid)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # **🔹 Deteksi MobileNet SSD**
        else:
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 15:
                        x1, y1, x2, y2 = map(int, detections[0, 0, i, 3:7] * [w, h, w, h])
                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        detected_centroids.append(centroid)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # **🔹 Tracking ID dengan Hungarian Algorithm**
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
                    # Deteksi orang masuk/keluar berdasarkan Y (atas/bawah)
                    if prev_centroid[1] < line_position and detected_centroids[j][1] > line_position:
                        down_count += 1  # Masuk
                    elif prev_centroid[1] > line_position and detected_centroids[j][1] < line_position:
                        up_count += 1  # Keluar
                else:
                    # Deteksi orang masuk/keluar berdasarkan X (kiri/kanan)
                    if prev_centroid[0] < line_position and detected_centroids[j][0] > line_position:
                        down_count += 1  # Masuk dari kiri
                    elif prev_centroid[0] > line_position and detected_centroids[j][0] < line_position:
                        up_count += 1  # Keluar ke kiri

                new_tracked_objects[obj_id] = (detected_centroids[j], frame_count, entry_frame)

        
        for j, new_c in enumerate(detected_centroids):
            if j not in col_ind:
                new_tracked_objects[next_id] = (new_c, frame_count, frame_count)
                next_id += 1

        tracked_objects = new_tracked_objects

        # **🔹 Update UI Metrics**
        total_visitor = (down_count-up_count)
        total_people_placeholder.metric("👥 Total Orang", total_visitor)
        up_count_placeholder.metric("⬆️ Keluar", up_count)
        down_count_placeholder.metric("⬇️ Masuk", down_count)

         # **🔹 Notifikasi & Screenshot**
        if total_visitor == 5 and not alert_50_sent:
            message = "⚠️ Alert! Pengunjung RPTRA sudah mencapai 50% kapasitas /n Mohon petugas standby berjaga!!"
            st.toast(message, icon="⚠️")
            screenshot_path = "screenshot_50.png"
            cv2.imwrite(screenshot_path, frame)
            screenshot_placeholder.image(screenshot_path, caption="Alert 50% Capacity", use_container_width=True)
            send_telegram_message(message, screenshot_path)
            alert_50_sent = True

        elif total_visitor == 7 and not alert_75_sent:
            message = "🚨 Alert! Pengunjung RPTRA sudah mencapai 75% kapasitas"
            st.toast(message, icon="🚨")
            screenshot_path = "screenshot_75.png"
            cv2.imwrite(screenshot_path, frame)
            screenshot_placeholder.image(screenshot_path, caption="Alert 75% Capacity /n /n Mohon petugas standby berjaga keliling RPTRA!!", use_container_width=True)
            send_telegram_message(message, screenshot_path)
            alert_75_sent = True

        frame_resized = cv2.resize(frame, (800, 600))
        video_placeholder.image(frame_resized, channels="BGR")
        time.sleep(0.03)

    cap.release()
