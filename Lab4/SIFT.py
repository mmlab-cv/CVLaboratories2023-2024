import cv2
import numpy as np

NUM_FEATURES = 400

img_1 = cv2.imread('../material/box.png', 0)
img_2 = cv2.imread('../material/box_in_scene.png', 0)

img_obj = img_1.copy()
img_scene = img_2.copy()

# Step 1: detect SIFT features
sift = cv2.SIFT_create(NUM_FEATURES)
kp_obj, dsc_obj = sift.detectAndCompute(img_obj, None)
kp_scene, dsc_scene = sift.detectAndCompute(img_scene, None)

img_obj = cv2.drawKeypoints(
    img_obj, kp_obj, img_obj, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_scene = cv2.drawKeypoints(
    img_scene, kp_scene, img_scene, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Object', img_obj)
cv2.imshow('Scene', img_scene)
cv2.waitKey(0)

# Step 2: Matching descriptors between the two images
matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.match(dsc_obj, dsc_scene)

img3 = cv2.drawMatches(img_obj, kp_obj, img_scene, kp_scene,
                        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img3)
cv2.waitKey(0)

good_matches = list(filter(lambda x: x.distance < 150, matches))

img3 = cv2.drawMatches(img_obj, kp_obj, img_scene, kp_scene,
                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Good Matches', img3)
cv2.waitKey(0)

# Step 3: stitching
obj_pts = np.float32(
    [kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
scn_pts = np.float32(
    [kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography
H, masked = cv2.findHomography(obj_pts, scn_pts, cv2.RANSAC)

# Warp image
dst = cv2.warpPerspective(img_1, H, (img_2.shape[1], img_2.shape[0]))
cv2.imshow('warped', dst)
cv2.waitKey(0)

# Threshold the image to get only black colors
mask = cv2.inRange(dst, 0, 2)

# display image
cv2.imshow("Mask", mask)
cv2.waitKey(0)

# Stitch image using masking

img_2[mask==0] = dst[mask==0]

cv2.imshow("Result", img_2)
cv2.waitKey(0)
