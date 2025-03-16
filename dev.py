import streamlit as st
import cv2
import numpy as np
import time
import requests

# Konfigurasi UI Streamlit
st.set_page_config(page_title="People Counting CCTV", layout="wide")

st.title("RPTRA CCTV Monitoring System")
st.write("Tracking real-time movement using object detection.")

# **🔹 Konfigurasi Telegram**
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

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

# **🔹 UI Card Metrics**
col1, col2, col3 = st.columns(3)
total_people_placeholder = col1.empty()
up_count_placeholder = col2.empty()
down_count_placeholder = col3.empty()

# **🔹 Load Model MobileNet-SSD**
net = cv2.dnn.readNetFromCaffe('sample/deploy.prototxt', 'sample/mobilenet_iter_73000.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

previous_centroids = []
up_count = 0
down_count = 0

# **🔹 Stream Video**
video_placeholder = st.empty()
screenshot_placeholder = st.empty()

cap = cv2.VideoCapture("sample/test.mp4")
alert_50_sent = False
alert_75_sent = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset ke frame awal
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (500, 500)), 0.007843, (500, 500), 127.5)
    net.setInput(blob)
    detections = net.forward()

    centroids = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            # **🔹 Gambar bounding box**
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    line_position = int(h // 2)
    current_up = 0
    current_down = 0

    if previous_centroids:
        for centroid in centroids:
            for prev_centroid in previous_centroids:
                if abs(centroid[0] - prev_centroid[0]) < 50:
                    if prev_centroid[1] < line_position and centroid[1] > line_position:
                        current_up += 1
                    elif prev_centroid[1] > line_position and centroid[1] < line_position:
                        current_down += 1

    up_count += current_down
    down_count += current_up
    previous_centroids = centroids

    # **🔹 Tambahkan Garis dan Label di Video**
    cv2.line(frame, (0, line_position), (w, line_position), (0, 255, 255), 2)
    cv2.putText(frame, "Keluar", (w // 10 , line_position - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.putText(frame, "Masuk", (w // 10, line_position + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # **🔹 Update UI**
    total_people_placeholder.metric("👥 Total Orang", len(centroids))
    up_count_placeholder.metric("⬆️ Keluar", up_count)
    down_count_placeholder.metric("⬇️ Masuk", down_count)

    # **🔹 Notifikasi & Screenshot**
    if down_count == 5 and not alert_50_sent:
        message = "⚠️ Alert! Pengunjung RPTRA sudah mencapai 50% kapasitas"
        st.toast(message, icon="⚠️")
        screenshot_path = "screenshot_50.png"
        cv2.imwrite(screenshot_path, frame)
        screenshot_placeholder.image(screenshot_path, caption="Alert 50% Capacity", use_container_width=True)
        send_telegram_message(message, screenshot_path)
        alert_50_sent = True

    elif down_count == 7 and not alert_75_sent:
        message = "🚨 Alert! Pengunjung RPTRA sudah mencapai 75% kapasitas"
        st.toast(message, icon="🚨")
        screenshot_path = "screenshot_75.png"
        cv2.imwrite(screenshot_path, frame)
        screenshot_placeholder.image(screenshot_path, caption="Alert 75% Capacity", use_container_width=True)
        send_telegram_message(message, screenshot_path)
        alert_75_sent = True

    frame_resized = cv2.resize(frame, (800, 600))
    video_placeholder.image(frame_resized, channels="BGR")

    time.sleep(0.03)

cap.release()
