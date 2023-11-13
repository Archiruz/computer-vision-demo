import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('../miyu.png', 0)

# Thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Operasi dilasi
kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Operasi erosi
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Menampilkan gambar hasil dilasi dan erosi
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()