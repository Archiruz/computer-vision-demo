import cv2

img = cv2.imread('../miyu.png')
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

cv2.imshow('Original', img)
cv2.imshow('Noise Reduction', dst)
cv2.waitKey()
cv2.destroyAllWindows()