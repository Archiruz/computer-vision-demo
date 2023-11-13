import cv2

# Load gambar
image = cv2.imread('../miyu.png')

if image is not None:
    # Menampilkan channel hijau
    green_channel = image.copy()
    green_channel[:, :, 0] = 0 # Menghilangkan channel warna biru
    green_channel[:, :, 2] = 0 # Menghilangkan channel warna merah
    cv2.imshow('Green Channel Image', green_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found")