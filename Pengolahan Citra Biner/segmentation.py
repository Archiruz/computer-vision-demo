import cv2

# Membaca gambar
image = cv2.imread('../miyu.png', 0)

# Segmentasi dengan thresholding adaptif
adaptive_threshold = cv2.adaptiveThreshold(image, 255, 
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

# Menampilkan gambar hasil segmentasi
cv2.imshow('Adaptive Thresholding', adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()