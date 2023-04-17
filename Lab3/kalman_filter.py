import cv2
import numpy as np
import matplotlib.pyplot as plt

frame = np.ones((800,800,3),np.uint8) * 255
last_mes = current_mes = np.array((2,1), np.float32)
last_pre = current_pre = np.array((2,1), np.float32)

def mousemove(event, x, y, s, p):
    global frame, current_mes, last_mes, current_pre, last_pre
    last_pre = current_pre
    last_mes = current_mes
    current_mes = np.array([[np.float32(x)],[np.float32(y)]])

    kalman.correct(current_mes)
    current_pre = kalman.predict()

    lmx, lmy = [l.astype(int) for l in last_mes[:2]]
    lpx, lpy = [l.astype(int) for l in last_pre[:2]]
    
    cmx, cmy = [l.astype(int) for l in current_mes[:2]]   
    cpx, cpy = [l.astype(int) for l in current_pre[:2]]  
    
    cv2.line(frame, (lmx[0],lmy[0]),(cmx[0],cmy[0]),(0,200,0))
    cv2.line(frame, (lpx[0],lpy[0]),(cpx[0],cpy[0]),(0,0,200))
    cv2.putText(frame, "Measurement: ({:.1f}, {:.1f})".format(np.float(cmx), np.float(cmy)),
                (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
    cv2.putText(frame, "Prediction: ({:.1f}, {:.1f})".format(np.float(cpx), np.float(cpy)),
                (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))


cv2.namedWindow("Kalman")
cv2.setMouseCallback("Kalman", mousemove)

# Dynamic parameters(4/6) and measurement parameters (2)
# kalman = cv2.KalmanFilter(4,2)
# # H
# kalman.measurementMatrix =np.array([[1,0,0,0],[0,1,0,0]], np.float32)
# # A
# kalman.transitionMatrix = \
#     np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
# # w
# kalman.processNoiseCov =  np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
# # v
# kalman.measurementNoiseCov =np.array([[1,0],[0,1]], np.float32) * 1

kalman = cv2.KalmanFilter(6,2)
kalman.measurementMatrix = \
    np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0]],np.float32)
kalman.transitionMatrix = \
    np.array([
        [1,0,1,0,0.5,0],
        [0,1,0,1,0,0.5],
        [0,0,1,0,1,0],
        [0,0,0,1,0,1],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]], np.float32)
kalman.processNoiseCov = \
    np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]], np.float32) * 0.003
kalman.measurementNoiseCov = \
    np.array([
        [1,0],
        [0,1]], np.float32) * 1

while(True):
    cv2.imshow('Kalman', frame)
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()