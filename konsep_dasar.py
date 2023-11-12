# import cv2
# import numpy as np

# # Inisialisasi kamera
# cap = cv2.VideoCapture(0)

# while True:
#     # Baca frame dari kamera
#     ret, frame = cap.read()

#     # Periksa apakah frame telah berhasil dibaca
#     if not ret:
#         break

#     # 1. Blurring (Pemburaman)
#     blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

#     # 2. Filtering (Penyaringan)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
#     filtered_frame = cv2.filter2D(frame, -1, kernel)

#     # 3. Noise Reduction (Pengurangan Noise)
#     noise_reduced_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

#     # Tampilkan frame asli dan frame yang telah diolah dalam jendela terpisah
#     cv2.imshow("Original Frame", frame)
#     cv2.imshow("Blurred Frame", blurred_frame)
#     cv2.imshow("Filtered Frame", filtered_frame)
#     cv2.imshow("Noise Reduced Frame", noise_reduced_frame)

#     # Jika tombol "q" ditekan, keluar dari loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Tutup kamera dan jendela OpenCV
# cap.release()
# cv2.destroyAllWindows()

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
    reduced_resolution = cv2.resize(combined, (640, 480))

    # Tampilkan citra hasil pengolahan dengan resolusi yang lebih rendah
    cv2.imshow('Real-Time Image Processing', reduced_resolution)

    # Keluar dari loop jika tombol "q" ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
