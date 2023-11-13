import cv2

path = '../miyu.png'

img = cv2.imread(path)

window_name = 'Blurred'
ksize = (10, 10)

image = cv2.blur(img, ksize, cv2.BORDER_DEFAULT)

cv2.imshow(window_name, image)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()