# Gunakan base image Python
FROM python:3.11

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Buat virtual environment
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Gunakan virtual environment saat container berjalan
ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8501

# Jalankan aplikasi (sesuaikan dengan file entry point)
CMD ["python3.11", "-m", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
