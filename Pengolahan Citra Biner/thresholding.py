import cv2

# Membaca gambar
image = cv2.imread('../miyu.png', 0) # Membaca sebagai citra grayscale

# Thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Menampilkan gambar hasil thresholding
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()