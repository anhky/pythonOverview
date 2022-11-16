import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
from counter_time import PinkColor
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (apply_classifier, check_img_size, non_max_suppression, scale_coords, set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, select_device

def detect(w1, h1, source, weights, conf_thres = 0.25, iou_thres = 0.45, img_size = 640):
    agnostic_nms, no_trace, augment = False, False, True
    classes = start_coordinates = end_coordinates = None
    frame_count_people = 0
    trace = not no_trace
    # cudnn.benchmark = True
   
    # Initialize
    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    # for path, img, frame, vid_cap in dataset:
    cap=cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter("RVideo/13.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1, h1))
    while True:
        ret, frame = cap.read()
        if ret:
            # frameCopy = frame.copy()
            center, w = PinkColor.pinkColor(frame) 
            cv2.circle(frame, center, 5, (255, 255, 255), -1)
            img = letterbox(frame, 640, stride=32)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, model, img, frame)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = frame
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if names[int(cls)] == "person" or names[int(cls)] == "sports ball":
                            c1, c2 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            if names[int(cls)] == "person":
                                if center != None and w !=None:     
                                    start_coordinates = center[0]
                                    end_coordinates = center[0] 
                                    
                                    # if center[0] <= start_coordinates:
                                    #     start_coordinates =center[0]     
                                    #     cv2.putText(frame, "start ", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))    
                                    # else:
                                    #     end_coordinates = center[0] 
                                    #     cv2.putText(frame, "end ", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))  
                        
                                    if center[0] < int(w / 2):
                                        start_coordinates = center[0]
                                    else:
                                        end_coordinates = center[0] 
                                    # if center[0]< start_coordinates:
                                    #     start_coordinates= center[0]
                                    #     cv2.putText(frame, "start ", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
                                    # if center[0] >= end_coordinates:
                                    #     end_coordinates = center[0]
                                    #     cv2.putText(frame, "end ", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255)) 
        
            if center!=None and start_coordinates != None and end_coordinates != None:
                if (int(c1[0])>=int(start_coordinates) and int(c1[0])<=int(end_coordinates)):
                    frame_count_people +=1  
            start_coordinates = start_coordinates
            end_coordinates = end_coordinates
            t_player = frame_count_people/fps
            # show(frame, t_player)
            cv2.putText(frame, "TIME " + str(t_player), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))    
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            
            vid_writer.write(frame)
        else:
            break
            # cv2.destroyAllWindows()
    # print(data_pink)
def detect_ball(w, h, source, weights, conf_thres = 0.25, iou_thres = 0.45, img_size = 640):
    agnostic_nms, no_trace, augment = False, False, True
    classes = start_coordinates = end_coordinates = None
    trace = not no_trace
    
    # cudnn.benchmark = True
   
    # Initialize
    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1
    count = 0
    # for path, img, frame, vid_cap in dataset:
    cap=cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    a = 0
    b = 0
    vid_writer = cv2.VideoWriter("RVideo/bong1_2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    while True:
            
        ret, frame = cap.read()
        if ret:
            # frame = cv2.line(frame, (int(w/4), 0), (int(w/4), int(h)), (255, 0, 0), 5)
            frameCopy = frame.copy()
            img = letterbox(frame, 640, stride=32)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, model, img, frame)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = frame
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'                    
                        if names[int(cls)] == "sports ball":
                            c1, c2 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            
                            if int(w/4)< (c1[0] + c2[0])/2 < (w*1)/2:
                                a=1
                            # if (c1[0] + c2[0])/2 < (w*1)/2:
                            #     a=1
                            elif (c1[0] + c2[0])/2 > (w*1)/2:
                                b = 1

                            if a ==1 and b==1 and int(c1[0] + c2[0])/2 > (w*1)/2:
                                count+=1
                                a = b = 0

            cv2.putText(frame, "count " + str(count), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255)) 
            # cv2.putText(frame, "count " + str(a), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255)) 
            # cv2.putText(frame, "count " + str(b), (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255)) 
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            vid_writer.write(frame)
        else:
            break

def show(frame, t_player):
    cv2.putText(frame, "TIME " + str(t_player), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if key == ord("q"):
    #     break

if __name__ == '__main__':
    weights = "weights/yolov7-tiny.pt"
    path = "D:/fdsproject/FDS/yolov7/video/"
    files = os.listdir(path)
    source = "video/13.mp4"
    cap=cv2.VideoCapture(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PinkColor = PinkColor(cap)
    detect(w, h, source, weights)
    # try:
    # for file in files:
    #     file_name, file_extension = os.path.splitext(os.path.basename(file))
    #     print("file_extension", file_extension)
    #     if file_extension !=' .mp4':
    #         print("AAAAA", file_name)
    #         with torch.no_grad():
    #             source = path + file
    #             cap=cv2.VideoCapture(source)
    #             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #             PinkColor = PinkColor(cap)
    #             detect(w, h, source, weights)
    #             pass
    # except:
    #     pass
    
