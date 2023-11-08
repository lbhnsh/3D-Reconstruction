from PIL import Image
import requests
import cv2
import numpy as np
import os
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

path=input("Enter image path: ")
original_image = Image.open(path)
width=original_image.size[0]
height=original_image.size[1]
original_image = original_image.convert('RGB')

from transformers import AutoProcessor

# AutoProcessor loads a OneFormerProcessor for us, based on the checkpoint
processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

# Prepares image for the model
panoptic_inputs = processor(images=original_image, task_inputs=["panoptic"], return_tensors="pt")
for k,v in panoptic_inputs.items():
  print(k,v.shape)

processor.tokenizer.batch_decode(panoptic_inputs.task_inputs)

from transformers import AutoModelForUniversalSegmentation

model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

# forward pass
with torch.no_grad():
  outputs = model(**panoptic_inputs)

panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[original_image.size[::-1]])[0]
print(panoptic_segmentation.keys())


# Create Segmented Mask
def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    ax.axis('off')
    instances_counter = defaultdict(int)
    handles = []
    
    n_segments=1

    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        n_segments=n_segments+1

    for segment in panoptic_segmentation['segments_info']:
      segment_id = segment['id']
      segment_label_id = segment['label_id']
      segment_label = model.config.id2label[segment_label_id]
      # print(f"Segment ID: {segment_label_id}, Class Name: {segment_label}")
    
    panoptic_segmented_path="/home/labhansh/Open3D/MiDaS/input/your_segmented_image.png"
    
    plt.savefig(panoptic_segmented_path,bbox_inches='tight',pad_inches=0)
    plt.show()

draw_panoptic_segmentation(**panoptic_segmentation)

# CHANGE THE DIRECTORY PATH AS PER YOUR CHOICE BY CHANGING IT IN LINE 70