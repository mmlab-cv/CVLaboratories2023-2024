import cv2
import numpy as np

SAMPLING = 30
MAX_CORNERS = 100
QUALITY = 0.01
MIN_DISTANCE = 10
BLOCK_SIZE = 3
USE_HARRIS = False
K_HARRIS = 0.04
webcam = False


cap = cv2.VideoCapture("../material/Video2.mp4")

frame_index = 0
while cap.isOpened():
    # Read video
    ret, frame = cap.read()

    # If video end reached
    if not ret:
        break

    # Copy frame to draw features on top of it and convert to grayscale
    frame_copy = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Select GFF Features
    if frame_index % SAMPLING == 0:
        corners = cv2.goodFeaturesToTrack(
            frame_gray, 
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY,
            minDistance=MIN_DISTANCE,
            blockSize=BLOCK_SIZE,
            useHarrisDetector=USE_HARRIS,
            k=K_HARRIS
        )
    else:
        # Track GFF features with Lucas-Kanade optical flow
        corners, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_corners, None)

    # Plot keypoints
    int_corners = corners.astype(int)
    for i, corner in enumerate(int_corners):
        x, y = corner.ravel()
        color = np.float64([i, 2 * i, 255 - i])
        cv2.circle(frame_copy, (x, y), 3, color)

    # Copy values for next iteration 
    prev_frame, prev_corners = frame.copy(), corners

    # Plot results
    cv2.imshow('GFF', frame_copy)
    
    # Wait and exit if q is pressed
    if cv2.waitKey(1) == ord('q') or not ret:
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()