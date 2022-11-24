import cv2
import numpy as np
from datetime import datetime
cap = cv2.VideoCapture("video/15.mp4")
kernel = (7, 7)
lower = np.array([150, 120, 222])
upper = np.array([160, 255, 255])

def empty(a):
    pass
cv2.namedWindow("TrackBars")
cv2.createTrackbar("cVal", "TrackBars", 10, 40, empty)
cv2.createTrackbar("bSize", "TrackBars", 77, 255, empty)

def preprocessing(frame, value_BSize, cVal):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    imgBlurred = cv2.GaussianBlur(mask, kernel, 0)
    gaussC = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value_BSize,                             cVal)
    imgDial = cv2.dilate(gaussC, kernel, iterations=3)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)

    return imgDial


def getContours(imPrePro):
    contours, hierarchy = cv2.findContours(imPrePro, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(imgCon, cnt, -1, (0, 255, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.circle(imgCon, (cX, cY), 7, (0, 0, 0), -1)

#######################################################################################################
t0 = datetime.now()
while True:
    success, frame = cap.read()
    if not success:
        break

    cVal = cv2.getTrackbarPos("cVal", "TrackBars")
    bVal = cv2.getTrackbarPos("bVal", "TrackBars")
    value_BSize = cv2.getTrackbarPos("bSize", "TrackBars")
    cVal = 0
    bVal = 0
    value_BSize = bVal
    value_BSize = max(3, value_BSize)
    if (value_BSize % 2 == 0):
        value_BSize += 1
    
    frame = cv2.flip(frame, 1)
    imgCon = frame.copy()
    imPrePro = preprocessing(frame, value_BSize, cVal)
    getContours(imPrePro)
    cv2.imshow("Preprocessed", imPrePro)
    cv2.imshow("Original", imgCon)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
t1 = datetime.now()
print(t1-t0)