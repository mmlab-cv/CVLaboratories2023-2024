import numpy as np
import cv2

MAX_FRAMES = 1000
LEARNING_RATE = -1  # alpha
HISTORY = 200       # t
N_MIXTURES = 5    # K (number of gaussians)
BACKGROUND_RATIO = 0.1 # Gaussian threshold
NOISE_SIGMA = 1     
MOG_VERSION = 1

cap = cv2.VideoCapture("../material/Video.mp4")

if MOG_VERSION == 1:
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA)
elif MOG_VERSION == 2:
    fgbg = cv2.createBackgroundSubtractorMOG2()
else:
    print(f"Unknown MOG version {MOG_VERSION}")
    quit(0)


for i in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If video end reached
    if not ret:
        break

    fgmask = fgbg.apply(frame, LEARNING_RATE)

    if MOG_VERSION == 2:
        bg = fgbg.getBackgroundImage()
        cv2.imshow('bg', bg)

    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frame', frame)

    # Wait and exit if q is pressed
    if cv2.waitKey(10) == ord('q') or not ret:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()