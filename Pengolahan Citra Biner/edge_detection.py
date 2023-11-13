import cv2

# Membaca gambar
image = cv2.imread('../miyu.png', 0)

# Deteksi tepi dengan metode Canny
edges = cv2.Canny(image, 100, 200)

# Menampilkan gambar hasil deteksi tepi
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()