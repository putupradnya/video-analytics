import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort  # Import SORT Tracker

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Inisialisasi SORT Tracker
tracker = Sort()

# URL CCTV
STREAM_URL = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_GdAgung.stream/playlist.m3u8"

# Variabel Counting
people_count = {"masuk": 0, "keluar": 0}
tracked_objects = {}

# **Fungsi Deteksi & Tracking**
def detect_debug(frame):
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 0 and conf > 0.2:  # Turunkan threshold ke 0.2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def detect_people(frame, line_y):
    global people_count, tracked_objects
    results = model(frame, verbose=False)
    
    frame_h, frame_w, _ = frame.shape
    line_y = int(frame_h * line_y / 100)  # Konversi persen ke pixel

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 0 and conf > 0.3:  # Hanya deteksi orang
                detections.append([x1, y1, x2, y2, conf])  # Format untuk SORT

    # **Gunakan SORT untuk tracking**
    tracked_objects_list = tracker.update(np.array(detections))

    new_tracked_objects = {}

    for obj in tracked_objects_list:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Titik tengah

        # **Pastikan ID objek tetap sama**
        if obj_id in tracked_objects:
            prev_y = tracked_objects[obj_id]
            
            # **Cek jika objek melewati garis**
            if prev_y < line_y and cy >= line_y:
                people_count["masuk"] += 1
            elif prev_y > line_y and cy <= line_y:
                people_count["keluar"] += 1

        # **Update posisi terbaru objek**
        new_tracked_objects[obj_id] = cy

        # **Gambar bounding box & ID**
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Titik tengah

    # **Update daftar objek yang dilacak**
    tracked_objects = new_tracked_objects

    # **Gambar garis deteksi**
    cv2.line(frame, (0, line_y), (frame_w, line_y), (255, 0, 0), 2)
    cv2.putText(frame, "Garis Deteksi", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame

# **Fungsi Streaming CCTV**
def stream_cctv(line_pos, masuk_card, keluar_card, fps_card):
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        st.error("Gagal membuka stream CCTV. Coba periksa URL atau koneksi internet.")
        return

    stframe = st.empty()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            st.error("Stream berakhir atau terjadi error.")
            break

        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = detect_people(frame, line_pos)
        frame = detect_people(frame, line_pos)

        stframe.image(frame, channels="RGB", use_container_width=True)

        # **Hitung FPS**
        fps = 1.0 / (time.time() - start_time)

        # **Update Card Nilai**
        masuk_card.metric(label="👥 Orang Masuk", value=people_count["masuk"])
        keluar_card.metric(label="🚪 Orang Keluar", value=people_count["keluar"])
        fps_card.metric(label="⚡ FPS", value=f"{fps:.2f}")

        time.sleep(0.03)  # Delay supaya tidak lag

    cap.release()

# **Streamlit UI**
st.title("🎥 Live CCTV with YOLOv8 + SORT - Counting People")
st.write("Deteksi orang & hitung jumlah masuk/keluar dengan tracking.")

# **Slider untuk mengatur posisi garis**
line_pos = st.slider("Atur Posisi Garis (%)", 10, 90, 50)

# **Container untuk menampilkan hasil secara real-time**
col1, col2, col3 = st.columns(3)
masuk_card = col1.empty()
keluar_card = col2.empty()
fps_card = col3.empty()

# **Tombol Mulai**
if st.button("Mulai Streaming"):
    stream_cctv(line_pos, masuk_card, keluar_card, fps_card)
