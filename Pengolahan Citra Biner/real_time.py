import cv2

# Membuat objek untuk webcam
cap = cv2.VideoCapture(0)

while True:
    # Membaca citra dari webcam
    ret, frame = cap.read()
    
    # Mengonversi citra ke citra biner
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Menampilkan citra biner secara real-time
    cv2.imshow('Real-time Binary Image', binary_frame)
    
    # Tombol 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan webcam dan menutup jendela
cap.release()
cv2.destroyAllWindows()