import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords, set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load

weights = "weights/yolov7-tiny.pt"

def travelTime(model, device, half, frame, conf_thres = 0.25, iou_thres = 0.45):
    agnostic_nms, augment = False, True

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    img = letterbox(frame, 640, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms)

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
                if names[int(cls)] == "person":
                    c1, c2 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    return c1, c2
    return None, None

def setup(img_size = 640):
    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once

    return model, device, half
