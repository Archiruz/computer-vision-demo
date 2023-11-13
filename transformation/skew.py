import cv2
import numpy as np

# membaca gambar
img = cv2.imread('./miyu.png')

# mengambil tinggi dan lebar gambar
height,width = img.shape[:2]

# derajat condong dari sumbu X/Y
skewX = 0.3
skewY = 0.1

# membuat matriks
matrix = np.float64([[1,skewX,0],[skewY,1,0]])

# membuat gambar menjadi condong
skewed_img = cv2.warpAffine(img,matrix,(width,height))

# menampilkan gambar
cv2.imshow('skewed image',skewed_img)
cv2.imshow('original image', img)

cv2.waitKey(0)
cv2.destroyAllWindows