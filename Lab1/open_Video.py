import numpy as np
import cv2

## CAPTURE VIDEO FROM CAMERA

cap = cv2.VideoCapture('Video.mp4')   #cap = cv2.VideoCapture('open_Video.py')
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
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 0)
    # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
