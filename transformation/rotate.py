import cv2

img = cv2.imread("./miyu.png")
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('Original',img)
cv2.imshow("Rotated Image", rotated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()