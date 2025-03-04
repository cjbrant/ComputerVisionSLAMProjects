import cv2

# load images
image1 = cv2.imread('lambo1.jpg', None)
image2 = cv2.imread('lambo2.jpg', None)

# sift instance
sift = cv2.SIFT_create()

# flann settings
FLANN_INDEX_KDTREE = 1 
index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
search_param = dict(checks=500)

# flann instance
flann = cv2.FlannBasedMatcher(index_param, search_param)

# detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# all matches
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# lowe's ratio test to filter matches
good_matches = []

for n, m in matches:
	if n.distance < 0.75 * m.distance:
		good_matches.append(n)


# draw matches 
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

# show
cv2.imshow('flann feature matching', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
