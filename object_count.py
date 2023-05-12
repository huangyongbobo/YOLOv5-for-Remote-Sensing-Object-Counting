import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torch.utils.data import DataLoader
import numpy as np

from models.experimental import attempt_load
from utils.datasets import count_dataset
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


# object count by object detection
def count(save_img=False):
    image_path, weights, count_class, view_img, save_txt, imgsz = \
        opt.image_path, opt.weights, opt.count_class, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    dataset = count_dataset(image_path, count_class, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    AE = 0
    SE = 0
    for path, img, im0s, vid_cap, target_count in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        pred_count = 0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    label_class, label_confidence = label.split()
                    if label_class == count_class:
                        pred_count += 1

        AE += abs(pred_count - target_count)
        SE += (pred_count - target_count) * (pred_count - target_count)

    MAE = AE / len(dataset)
    RMSE = np.sqrt(SE / len(dataset))

    print('MAE: %.5f' % MAE)
    print('RMSE: %.5f' % RMSE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/exp6/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--image_path', type=str, default='./RSOC_small-vehicle/test/images', help='test image path')  # detect_image/images, 0 for webcam
    parser.add_argument('--count_class', type=str, default='small-vehicle', help='count class')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                count()
                strip_optimizer(opt.weights)
        else:
            count()
