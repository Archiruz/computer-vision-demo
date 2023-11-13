import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./miyu.png")
# Loading the image

half = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
taller = cv2.resize(image, (1050, 1610))

stretch_near = cv2.resize(image, (780, 540), 
                            interpolation=cv2.INTER_LINEAR)


Titles = ["Original", "Half", "Taller", "Interpolation Nearest"]
images = [image, half, taller, stretch_near]
count = 4

for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

plt.show()
