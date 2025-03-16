# People Counting Project Demo

Aplikasi **Traffic Counting** berbasis **Python & Streamlit** untuk mendeteksi dan menghitung jumlah orang  yang masuk dan keluar dari suatu area menggunakan **YOLOv8 dan MobileNet SSD**.

## 📥 Instalasi

### 1️⃣ Clone Repository

```bash
git clone https://github.com/username/traffic-counting-yolo.git
cd traffic-counting-yolo
```

### 2️⃣ Buat Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## Cara Penggunaan

### 🔹 Jalankan Aplikasi Streamlit

```bash
streamlit run dashboard.py

atau

streamlit run aws-dashboard.py
```

### 🔹 Input Video

- Bisa input **URL video** atau **upload file video**
- Sistem akan otomatis mendeteksi dan menghitung objek

---

## ⚙️ Konfigurasi & Optimalisasi

### 1️⃣ Gunakan Hardware Acceleration (MPS untuk Mac M1/M2)

Tambahkan di awal kode:

```python
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model.to(device)
```

### 2️⃣ Ubah Threshold Confidence & IoU

```python
model.predict(frame, conf=0.5, iou=0.7)
```

- **conf (Confidence Threshold):** Objek terdeteksi hanya jika confidence > 50%
- **iou (Intersection over Union):** Menghindari prediksi ganda

### 3️⃣ Ubah Ukuran Frame untuk Performa Lebih Baik

```python
frame = cv2.resize(frame, (640, 384))
```

Ukuran lebih kecil = **Inference lebih cepat** ⚡

---

Selamat mencoba! 🚀
