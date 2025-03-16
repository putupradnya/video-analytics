import cv2

def process_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka stream CCTV.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream berakhir atau error.")
            break

        # Tampilkan frame dalam jendela (opsional)
        cv2.imshow('CCTV Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ganti dengan URL CCTV Anda
stream_url = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8"
process_stream(stream_url)
