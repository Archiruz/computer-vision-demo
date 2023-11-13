import cv2
import numpy as np

# membaca gambar
img = cv2.imread('./miyu.png')

# mengambil tinggi dan lebar gambar
height,width = img.shape[:2]

# jumlah pixel yang digeser
translationX = 100
translationY = 100
# membuat matriks
matrix = np.float64([[1,0,translationX],[0,1,translationY]])

# membuat gambar menjadi berpindah
translated_img = cv2.warpAffine(img,matrix,(width,height))

# menampilkan gambar
cv2.imshow('Translated image',translated_img)
cv2.imshow('Original image', img)

cv2.waitKey(0)
cv2.destroyAllWindows