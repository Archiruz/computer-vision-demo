import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    if not ret:
        break

    # Blurring citra dengan Gaussian Blur
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # Filter citra dengan filter kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(frame, -1, kernel)

    # Pengurangan noise menggunakan filter median
    denoised = cv2.medianBlur(frame, 5)

    # Susun hasil pengolahan citra dalam window 2x2
    top_row = np.hstack((frame, blurred))
    bottom_row = np.hstack((filtered, denoised))
    combined = np.vstack((top_row, bottom_row))

    # Kurangi resolusi citra hasil pengolahan
    reduced_resolution = cv2.resize(combined, (1280, 960))

    # Tampilkan citra hasil pengolahan dengan resolusi yang lebih rendah
    cv2.imshow('Real-Time Image Processing', reduced_resolution)

    # Keluar dari loop jika tombol "q" ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()