import cv2
import numpy as np  
import logging

start_coordinates = 0
end_coordinates = 0

class PinkColor:
    def __init__(self, camera) -> None:

        #Define the threshold for finding a blue object with hsv
        # lower = np.array([150, 170, 222])
        # upper = np.array([160, 255, 255])
        lower = np.array([150, 20, 222])
        upper = np.array([160, 255, 255])

        w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS)

        self.lower = lower
        self.upper = upper
        self.w = w
        self.h = h
        self.frame_count = frame_count

    def pinkColor(self, frame):
        # frame = frame[int(self.h * 1 / 5):int(self.h * 3 / 5), 0:self.w] 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # if radius >= 0:
            #     cv2.circle(result, (int(x), int(y)), int(radius),
            #             (0, 255, 255), 2)
            #     cv2.circle(result, center, 5, (0, 0, 255), -1)

            return center, self.w
        return None, None

if __name__ == "__main__":
    # VIDEO: 1, 2, 3.mp4
    cap=cv2.VideoCapture("1.mp4")
    start_coordinates = 0
    end_coordinates = 0
    Ball = PinkColor(cap)
    while True:
        ret, frame = cap.read()
        framePinkColor, start_coordinates, end_coordinates = Ball.pinkColor(frame)
        if start_coordinates != 0:
            start = start_coordinates
        if end_coordinates != 0:
            end = end_coordinates

        cv2.imshow('image', framePinkColor)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


