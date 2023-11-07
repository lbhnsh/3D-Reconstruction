# CHANGE LINE 8, 10, 13, 17, 40, 45 ACCORDING TO YOUR DESIRED PATH NAMES

import cv2
import numpy as np
import os

# Load image using OpenCV
original_image= cv2.imread("/home/labhansh/Downloads/room3.png")
original_image = cv2.resize(original_image,(original_image.shape[1],original_image.shape[0]))
instance_image = cv2.imread('/home/labhansh/Open3D/MiDaS/input/room_segment.png') 
instance_image = cv2.resize(instance_image,(original_image.shape[1],original_image.shape[0]))
instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
depth_image= cv2.imread("/home/labhansh/Open3D/MiDaS/depthmap_segments/labhansh_home.png")
depth_image = cv2.resize(depth_image,(original_image.shape[1],original_image.shape[0]))


n_segments = 5 # Replace with the desired number of segments

# Reshape the image to a 2D array of pixels
pixels = instance_image.reshape((-1, 3)).astype(np.float32)

# Perform k-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, n_segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reshape the labels to match the original image shape
labels = labels.reshape(instance_image.shape[:2])

# Loop through the segments and save each one as a separate image
for segment_id in range(n_segments):
    # Create a mask for the current segment
    mask = (labels == segment_id).astype(np.uint8)

    # Apply the mask to the original image
    # segmented_rgb_image = cv2.bitwise_and(instance_image, instance_image, mask=mask)

    segmented_rgb_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    segmented_depth_image=cv2.bitwise_and(depth_image, depth_image, mask=mask)
    # Save the segmented image to a file
    output_dir1 = '/home/labhansh/Open3D/segmented_rgb_images'
    os.makedirs(output_dir1, exist_ok=True)
    output_file = os.path.join(output_dir1, f'segment_rgb_{segment_id}.png')
    cv2.imwrite(output_file, segmented_rgb_image)
    # saving depth segments
    output_dir2 = '/home/labhansh/Open3D/segmented_depth_images'
    os.makedirs(output_dir2, exist_ok=True)
    output_file = os.path.join(output_dir2, f'segment_depth_{segment_id}.png')
    cv2.imwrite(output_file, segmented_depth_image)


