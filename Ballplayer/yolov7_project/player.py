import detectyolo
import detectMilestone
import cv2
from datetime import datetime

def run2():
    
    cap=cv2.VideoCapture("video/1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    model, device, half = detectyolo.setup()
    frame_count_people = 0
    t0 = datetime.now()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        data = detectMilestone.detectColor(frame.copy())
        c1, c2 = detectyolo.travelTime(model, device, half, frame.copy())
        if len(data)>0:
            data= [data[0], data[-1]]
            cv2.putText(frame, "end", data[1], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 250))
            cv2.putText(frame, "start", data[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 250))
            cv2.circle(frame, data[0], 7, (0, 0, 0), -1)
            cv2.circle(frame, data[1], 7, (0, 0, 0), -1)
        if len(data)>0 and c1 !=None:
            # print(c1, c2)
            centerX = (c1[0] + c2[0])/2
            if centerX >= data[0][0] and centerX <= data[1][0]:
                frame_count_people +=1  

        t_player = frame_count_people/fps
        cv2.putText(frame, "TIME " + str(t_player), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    t1 = datetime.now()
    print(t1-t0)
if __name__ == "__main__":
    run2()