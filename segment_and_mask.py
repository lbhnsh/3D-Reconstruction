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


'''
# COMMENT LATER
class_names = model.config.id2label
# Print or display the available classes
for class_id, class_name in class_names.items():
    print(f"Class ID: {class_id}, Class Name: {class_name}")
'''

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

    instance_image = cv2.imread(panoptic_segmented_path) 
    instance_image = cv2.resize(instance_image,(width,height))
    instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
    original_image= cv2.imread(path)
    original_image = cv2.resize(original_image,(width,height))
    # Reshape the image to a 2D array of pixels
    pixels = instance_image.reshape((-1, 3)).astype(np.float32)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape the labels to match the original image shape
    labels = labels.reshape(instance_image.shape[:2])

    # Loop through the segments and save each one as a separate image
    # for segment1_id in range(n_segments):
    for segment in panoptic_segmentation['segments_info']:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        # Create a mask for the current segment
        mask = (labels == segment_id).astype(np.uint8)
        # Apply the mask to the original image
        segmented_rgb_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        plt.imshow(segmented_rgb_image)
        plt.show()
        # Save the segmented image to a file
        
        if segment_label_id in [13, 56, 57, 59, 60, 72, 104, 120, 121]:
          print(segment_label)
          output_dir1 = '/home/labhansh/Open3D/segmented_rgb_images'
          os.makedirs(output_dir1, exist_ok=True)
          output_file = os.path.join(output_dir1, f'segment_rgb_{segment_label_id}_{segment_id}.png')
          cv2.imwrite(output_file, segmented_rgb_image)

    return n_segments

n_segments=draw_panoptic_segmentation(**panoptic_segmentation)

