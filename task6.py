import cv2
import numpy as np

image_path = r'C:\Users\ETHRIJIT\Desktop\task6\image.jpg'

# Load the image
image = cv2.imread(image_path)

# Resizing to 224x224
resized_image = cv2.resize(image, (224, 224))

# Grayscale conversion
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Cropping to extract only the dog region (example coordinates)
x, y, w, h = 50, 50, 100, 100  # Example coordinates, adjust according to your image
cropped_image = resized_image[y:y+h, x:x+w]

# Rotation by 45 degrees
rows, cols = cropped_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (cols, rows))

# Flip left to right
flipped_image = cv2.flip(rotated_image, 1)

# Gaussian blurring
gaussian_blurred_image = cv2.GaussianBlur(flipped_image, (5, 5), 0)

# Median blurring
median_blurred_image = cv2.medianBlur(flipped_image, 5)

# Edge detection using Canny
canny_edges = cv2.Canny(flipped_image, 100, 200)

# Background subtraction (Example using MOG2)
background_subtractor = cv2.createBackgroundSubtractorMOG2()
foreground_mask = background_subtractor.apply(flipped_image)

# Display the results
cv2.imshow('Original Image', resized_image)
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Flipped Image', flipped_image)
cv2.imshow('Gaussian Blurred Image', gaussian_blurred_image)
cv2.imshow('Median Blurred Image', median_blurred_image)
cv2.imshow('Canny Edges', canny_edges)
cv2.imshow('Background Subtraction', foreground_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
