import cv2 
import time
import argparse
import os


# For HOG plot
from skimage import exposure
from skimage import feature

WINSIZE = (64, 128) # Must be this size because of the people detector
BLOCKSIZE = (16, 16)
BLOCKSTRIDE = (8, 8)
CELLSIZE = (8, 8)
BINS = 9
SPEED = 'slow'


cap = cv2.VideoCapture("../material/hog1.mp4")

# Initialise multiscale HOG detector
hog = cv2.HOGDescriptor(
    WINSIZE, 
    BLOCKSIZE, 
    BLOCKSTRIDE, 
    CELLSIZE, 
    BINS
)


# Initialize people detector ((64, 128) default window)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if (cap.isOpened() == False):
    print('Error while trying to open video. Please check again...')

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    hogImage = frame.copy()
    if ret == True:
        start_time = time.time()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if SPEED == 'fast':
            rects, weights = hog.detectMultiScale(img_gray, padding=(4, 4), scale=1.02)
        elif SPEED == 'slow':
            rects, weights = hog.detectMultiScale(img_gray, winStride=(4, 4), padding=(4, 4), scale=1.02)
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.13:
                continue
            elif weights[i] < 0.3 and weights[i] > 0.13:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            if weights[i] < 0.7 and weights[i] > 0.3:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 122, 255), 2)
            if weights[i] > 0.7:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        cv2.putText(frame, 'High confidence', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Moderate confidence', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 122, 255), 2)
        cv2.putText(frame, 'Low confidence', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Compute HOG features for visualisation
        (H, hogImage) = feature.hog(hogImage, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualize=True,channel_axis=-1)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(50, 255))
        hogImage = hogImage.astype("uint8")

        cv2.imshow("HOG Image", hogImage)
        cv2.imshow("HOG Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break