import cv2

# load images
image1 = cv2.imread('lambo1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('lambo2.jpg', cv2.IMREAD_GRAYSCALE)


# sift detector
sift = cv2.SIFT_create()

# map keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


# check descriptors exist
if descriptors1 is None or descriptors2 is None:
	print("Error: no descriptors found in one or both images")
	exit()

# brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# match descriptors and sort 
matches =bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None)

# show
cv2.imshow('Feature Matching (SIFT)', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
