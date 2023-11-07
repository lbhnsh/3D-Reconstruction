import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# depth_map = cv2.bitwise_and('/home/labhansh/Open3D/segmented_depth_images/segment_depth_10.png', '/home/labhansh/Open3D/segmented_depth_images/segment_depth_10.png', mask='/home/labhansh/Open3D/segmented_rgb_images/segment_depth_10.png')
depth_map = cv2.imread('/home/labhansh/Open3D/segmented_depth_images/segment_depth_1.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Depth-Based Segmentation', depth_map)

count=0
sum=0
for i in range(500):
    for j in range(500):
        if depth_map[i][j]!=0:
            count=count+1
            sum=sum+depth_map[i][j]
avg=sum/count
print(count)
print(avg)
cv2.waitKey(0)
cv2.destroyAllWindows()


