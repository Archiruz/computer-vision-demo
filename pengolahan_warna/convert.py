import cv2

# Load gambar
image = cv2.imread('../miyu.png')

if image is not None:
    # Konversi dari RGB ke Grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found")