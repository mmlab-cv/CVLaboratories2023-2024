import numpy as np
import cv2


def bg_update(current_frame, prev_bg, alpha):
    bg = alpha * current_frame + (1 - alpha) * prev_bg

    # print(f"Background dtype (before): {bg.dtype}")
    # print(f"Background dtype (after): {bg.dtype}")

    bg = np.uint8(bg)
    
    return bg

background = None
MAX_FRAMES = 1000
THRESH = 50
MAXVAL = 255
ALPHA = 0.05

cap = cv2.VideoCapture("../material/Video.mp4")
    

for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If video end reached
    if not ret:
        break

    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)    
    
    if t == 0:
        # Train background with first frame
        background = frame_gray
    else:
        # Background subtraction
        diff = cv2.absdiff(background, frame_gray)

        # Mask thresholding
        ret, motion_mask = cv2.threshold(diff, THRESH, MAXVAL, cv2.THRESH_BINARY)

        # Update background
        # background = bg_update(frame_gray, background, alpha = ALPHA)

        # Display the resulting frames
        cv2.imshow('Frame', frame)
        cv2.imshow('Motion mask', motion_mask)
        cv2.imshow('Background', background)
        
        # Wait and exit if q is pressed
        if cv2.waitKey(1) == ord('q') or not ret:
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()