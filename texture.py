import cv2
import numpy as np
from skimage.feature import graycomatrix
from skimage import io, color, img_as_ubyte

# Load the image
image_path = 'texture_image.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to uint8 for GLCM computation
gray_image_uint8 = img_as_ubyte(gray_image)

# Define GLCM properties
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['energy', 'homogeneity', 'correlation']

# Compute GLCM
glcm = graycomatrix(gray_image_uint8, distances=distances, angles=angles, symmetric=True, normed=True)

# Texture segmentation based on GLCM properties
segmented_image = np.zeros_like(gray_image)

for prop in properties:
    threshold = 0.5  # You can adjust the threshold based on your specific image and requirements
    property_values = glcm[:, :, 0, properties.index(prop)]  # Use the correct index for the property
    mask = property_values > threshold
    segmented_image[mask] = 255

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
