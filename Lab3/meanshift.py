import numpy as np
import cv2

from matplotlib import pyplot as plt

webcam = False
PLOT_HIST = True
USE_CAMSHIFT = True

if webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("../material/Video2.mp4")

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = cv2.selectROI('Frame', frame, showCrosshair=False)
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (x, y, w, h)

# Set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Histogram plotting
if PLOT_HIST:
    hist_full = cv2.calcHist([frame], [0], None, [256], [0,256])
    plt.subplot(221), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.subplot(222), plt.imshow(mask, cmap='gray')
    plt.subplot(223), plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.subplot(224), plt.plot(hist_full), plt.plot(roi_hist)
    plt.xlim([0, 256])
    plt.show()

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret ,frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # apply meanshift to get the new location
    if USE_CAMSHIFT:
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret).astype(int)
        frame = cv2.polylines(frame, [pts], True, 255, 2)
    else:
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('dst', dst)

    # Wait and exit if q is pressed
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
