
import cv2
import sys
import numpy as np

## OPEN, DISPLAY, SAVE an image
img = cv2.imread('material\home.jpg')
if img is None:
    sys.exit("Could not read the image.")
cv2.imshow("Display window", img)
k = cv2.waitKey(0)
if k == ord("s"):
    cv2.imwrite("material\image_2.jpg", img)

