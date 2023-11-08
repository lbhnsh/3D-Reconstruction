from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8m-seg.pt")
predict = model.predict("/home/mrudul/Example/Separate_example.jpg" , save = True , save_txt = True)

# print(predict[0].masks.masks[0])