import cv2

# load images
image1 = cv2.imread('image1.jpg', None)
image2 = cv2.imread('image2.jpg', None)

# sift instance
sift = cv2.SIFT_create()

# flann settings
FLANN_INDEX_KDTREE = 1 
index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=500)
search_param = dict(checks=5000)

# flann instance
flann = cv2.FlannBasedMatcher(index_param, search_param)

# detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# check keypoints
MIN_KEYPOINTS = 10
if len(keypoints1) < MIN_KEYPOINTS or len(keypoints2) < MIN_KEYPOINTS:
    print(f"Error: Not enough keypoints detected. Image1: {len(keypoints1)}, Image2: {len(keypoints2)}")
    exit()

# check descriptors
MIN_KEYPOINTS = 10
if len(keypoints1) < MIN_KEYPOINTS or len(keypoints2) < MIN_KEYPOINTS:
    print(f"Error: Not enough keypoints detected. Image1: {len(keypoints1)}, Image2: {len(keypoints2)}")
    exit()


# all matches
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# lowe's ratio test to filter matches
good_matches = []

for n, m in matches:
	if n.distance < 0.75 * m.distance:
		good_matches.append(n)


# draw matches 
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:50], None)

# show
cv2.imshow('flann feature matching', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
