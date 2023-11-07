import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as od
import pandas as pd
import PIL
import tensorflow as tf
from ultralytics import YOLO
import cv2

def tensor_to_image(tensor):
    tensor = tensor *255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def create_mask(predictions, c_id, confidence_threshold=0.5):
    conf_scores = predictions[...,c_id]  # Confidence scores
    class_probs = predictions[...,c_id]  # Class probs
    objectness_score = conf_scores * class_probs.max(dim=-1).values
    if c_id is not None:
        # Filter by class
        class_mask = (class_probs.argmax(dim=-1) == c_id).float()
        mask = (objectness_score >= confidence_threshold) * class_mask
    else:
        # Filter without considering class
        mask = (objectness_score >= confidence_threshold)
    return mask
    


model = YOLO("yolov8m-seg.pt")
predict = model.predict("/home/param/Documents/dog.jpg", save = True, save_txt = True)
output = od.io.read_image("/home/param/Documents/runs/segment/predict/dog.jpg")
plt.imshow(output)
plt.show()
print(predict[0])
print(predict[0].masks[0])
# print(predict[0].shape)
# print(len(predict))
# print(predict[0].masks.shape)
# print(type(predict[0]))
# #cv2.imshow((predict[0].masks[0].unique() * 255 ).astype("uint8"))
# #seg_image = predict[0].masks.numpy()
seg_image = predict[0].masks[0] * 255
cv2.imwrite("/home/param/Documents/runs/segment/predict/mask1.jpg", seg_image)
#masked_image=od.io.read_image(predict[0].masks.numpy()*255).astype("uint8")
masked_image = od.io.read_image(seg_image)
plt.imshow(masked_image)
plt.show()
# mask = create_mask(predict, confidence_threshold=0.5, c_id=0)

