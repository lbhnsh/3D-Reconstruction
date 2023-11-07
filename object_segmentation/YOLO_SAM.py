import supervision as sv
import torch
import os
import math
from numpy import random
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)
