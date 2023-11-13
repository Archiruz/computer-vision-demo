import cv2
import numpy as np

# Load gambar
image = cv2.imread('../miyu.png')

if image is not None:
    # Konversi gambar ke skema warna HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tentukan range warna biru muda dalam HSV
    lower_light_blue = np.array([160, 50, 50])
    upper_light_blue = np.array([180, 255, 255])

    # Buat mask untuk objek dengan warna merah
    mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('Color Detection Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found")