import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# import midas_implement as mid


# Load your RGB image
rgb_image = cv2.imread("/home/labhansh/Open3D/MiDaS/input/beach_orignal.png")
segmented_mask = cv2.imread('/home/labhansh/Open3D/MiDaS/input/beach_panoptic.png')
# plt.imshow(segmented_mask)
# plt.show()
# plt.imshow(rgb_image)
# plt.show

# Find contours in the mask
contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''
# Extract and Process Segmented Regions: COPY 
segmented_regions = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    segmented_region = rgb_image[y:y+h, x:x+w]
    segmented_regions.append(segmented_region)
'''
# Extract and Process Segmented Regions:
segmented_regions = [], depth_regions =[]
count=1
for contour in contours:
    
    x, y, w, h = cv2.boundingRect(contour)
    segmented_region = rgb_image[y:y+h, x:x+w]
    output_path = f"/home/labhansh/Open3D/MiDaS/input_segments/segment{count}.png"  # Replace with your desired path and file format
    cv2.imwrite(output_path, (segmented_region))
    count=count+1

    depth_region = 
    segmented_regions.append(segmented_region)

# Create Point Clouds from Segmented Regions with Open3D:
segmented_point_clouds = []

for segmented_region in segmented_regions:
    # Assuming you have depth information corresponding to this region
    # Create an RGBD image for the segmented region
    segmented_rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
        segmented_region, depth_region)

    # Create a point cloud for the segmented region
    segmented_point_cloud = o3d.geometry.create_point_cloud_from_rgbd_image(
        segmented_rgbd_image, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    segmented_point_clouds.append(segmented_point_cloud)

# Combined Point Cloud
combined_point_cloud = o3d.geometry.PointCloud()

for segmented_point_cloud in segmented_point_clouds:
    combined_point_cloud += segmented_point_cloud

# Visualize or further process the combined point cloud
o3d.visualization.draw_geometries([combined_point_cloud])




'''
rgb_image = o3d.io.read_image(mid.input_path)
depth_image = o3d.io.read_image(mid.output_path)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image)

output_path = "/home/labhansh/Open3D/MiDaS/output/dogs_depth.png"  # Replace with your desired path and file format
cv2.imwrite(output_path, (output))
'''
import tensorflow
