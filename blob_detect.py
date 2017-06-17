import cv2
import numpy as np

d = cv2.imread('post-morph.jpg')
im = cv2.imread('images/DJI_0273.JPG')
params = cv2.SimpleBlobDetector_Params()
# Detect circles
params.filterByCircularity = True
params.minCircularity = 0.15
params.maxCircularity = 0.8
# Threshold for splitting images
params.minThreshold = 10
params.maxThreshold = 500
# filter by color
params.filterByColor = False
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
# Filter by Inertia
params.filterByArea = True
params.minArea = 100

params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 0.5
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector(params)
# Detect blobs.
keypoints = detector.detect(d)
# Draw detected blobs as circles.
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("keypoints", im_with_keypoints)
cv2.imwrite('post-blob2.jpg', im_with_keypoints)
cv2.waitKey(0)