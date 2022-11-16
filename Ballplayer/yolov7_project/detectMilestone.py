import cv2
import numpy as np
import operator

kernel = (7, 7)
lower = np.array([150, 120, 222])
upper = np.array([160, 255, 255])

def preprocessing(frame, value_BSize, cVal):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask,(5,5),0)
    # mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    imgBlurred = cv2.GaussianBlur(mask, kernel, 4)
    gaussC = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value_BSize, cVal)
    imgDial = cv2.dilate(gaussC, kernel, iterations=3)
    # imgErode = cv2.erode(imgDial, kernel, iterations=1)

    return imgDial


def getContours(imPrePro):
    contours, hierarchy = cv2.findContours(imPrePro, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = [cX, cY]
            data.append(center)
    data = sorted(data, key= (operator.itemgetter(0)))
    return data


def detectColor(frame):
    cVal = 0
    bVal = 40
    value_BSize = bVal
    value_BSize = max(3, value_BSize)
    if (value_BSize % 2 == 0):
        value_BSize += 1
    imPrePro  = preprocessing(frame, value_BSize, cVal)
    data = getContours(imPrePro)

    return data

def show():
    cap=cv2.VideoCapture("video/1.mp4")
    start = end = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cVal = 0
        bVal = 0
        value_BSize = bVal
        value_BSize = max(3, value_BSize)
        if (value_BSize % 2 == 0):
            value_BSize += 1
        imgCon = frame.copy()
        imPrePro  = preprocessing(frame, value_BSize, cVal)
        data = getContours(imPrePro)
        if len(data)>0:
            data= [data[0], data[-1]]
            cv2.putText(imgCon, "end", data[1], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 250))
            cv2.putText(imgCon, "start", data[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 250))
        cv2.imshow("Frame", imgCon)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# if __name__ == "__main__":
#     show()

