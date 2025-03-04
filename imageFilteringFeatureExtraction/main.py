import cv2
import numpy as np

# Load image
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(image, 50, 150)

# Show images
cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.imshow('Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
