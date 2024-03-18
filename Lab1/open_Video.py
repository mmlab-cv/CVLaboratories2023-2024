import numpy as np
import cv2

## CAPTURE VIDEO FROM CAMERA

cap = cv2.VideoCapture(0)   #cap = cv2.VideoCapture('Video.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
# Capture frame-by-frame
     ret, frame = cap.read()
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # # Our operations on the frame come here
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     # Display the resulting frame
     cv2.imshow('frame', gray)
     if cv2.waitKey(1) == ord('q'):
         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

## SAVE VIDEO
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.mkv', cv2.CAP_FFMPEG, fourcc, 20.0, (width,  height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 0)
    # write the flipped frame
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
