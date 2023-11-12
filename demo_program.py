import cv2
import argparse
import numpy as np

def edge_detection_segmentation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)
    cv2.imshow('Edge Detection Segmentation', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_based_segmentation(image_path, lower_range, upper_range):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color = np.array(lower_range)
    upper_color = np.array(upper_range)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('Color-Based Segmentation', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def texture_based_segmentation(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Metode GLCM (contoh)
    glcm = cv2.imgproc.text.CV_GLCM()
    glcm.set_images(gray, gray)
    contrast = glcm.compute(0, cv2.imgproc.text.CV_GLCM_CONTRAST)

    # Ambil piksel dengan kontrast tertinggi
    _, max_val, _, max_loc = cv2.minMaxLoc(contrast)
    mask = np.zeros_like(gray)
    mask[max_loc] = 255

    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('Texture-Based Segmentation', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def intensity_based_segmentation(image_path, threshold_value):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow('Intensity-Based Segmentation', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour_based_segmentation(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar kontur pada citra asli
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Contour-Based Segmentation', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Segmentation Methods")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--method", type=str, choices=["edge", "color", "texture", "intensity", "contour"],
                        help="Segmentation method to apply")

    args = parser.parse_args()

    if args.method == "edge":
        edge_detection_segmentation(args.image_path)
    elif args.method == "color":
        # Example color range for blue
        color_lower_range = [100, 100, 100]
        color_upper_range = [140, 255, 255]
        color_based_segmentation(args.image_path, color_lower_range, color_upper_range)
    elif args.method == "texture":
        texture_based_segmentation(args.image_path)
    elif args.method == "intensity":
        # Example threshold value
        threshold_value = 128
        intensity_based_segmentation(args.image_path, threshold_value)
    elif args.method == "contour":
        contour_based_segmentation(args.image_path)
    else:
        print("Invalid segmentation method. Choose from 'edge', 'color', 'texture', 'intensity', 'contour'.")
