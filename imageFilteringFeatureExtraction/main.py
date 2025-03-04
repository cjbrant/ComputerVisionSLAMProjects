import cv2
import numpy as np

# load image
image = cv2.imread('input.jpg')

# check if grey and if not convert
if len(image.shape) == 2:
	grey = image
else:
	grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur
blurred = cv2.GaussianBlur(grey, (7, 7), 0)

# apply canny edge detection
# using adaptive thresholds, high=3*low
medianVal = np.median(grey)
lowThresh = int(0.66 * medianVal)
highThresh = int(1.33 * medianVal)

edges = cv2.Canny(grey, lowThresh, highThresh)

# apply Harris corner detection
harris = cv2.cornerHarris(grey, blockSize=2, ksize=3, k=0.04)

# dilation if needed
#harris = cv2.dilate(harris, None)

# SIFT detector
sift = cv2.SIFT_create()

# detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(grey, None)

# draw keypoints
sift_image = cv2.drawKeypoints(image, keypoints, None)

# ORB (faster than SIFT)
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(grey, None)
orb_image = cv2.drawKeypoints(image, keypoints, None)



# show images
cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.imshow('Edges', edges)

# mark Harris corners in red
image[harris > 0.01 * harris.max()]=[0,0,255]

cv2.imshow('Harris Corners', image)
cv2.imshow('SIFT Features', sift_image)
cv2.imshow('ORB Features', orb_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

