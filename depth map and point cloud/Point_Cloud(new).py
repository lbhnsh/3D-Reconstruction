import open3d as od
import numpy as np
import cv2
import matplotlib.pyplot as plt
rgb_image = od.io.read_image("/home/param/Documents/dog.jpg")
depth_image = od.io.read_image("/home/param/documents/Figure_1.png")
rgbd_image = od.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image)
pcd = od.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    od.camera.PinholeCameraIntrinsic(
        od.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
od.visualization.draw_geometries([pcd])

# using a transformation matrix in point cloud generation allows you to accurately convert depth information 
# from 2D image space to 3D world space and ensures that the point cloud is aligned and registered correctly with 
# other data sources. This is crucial for various applications like 3D reconstruction, object detection, and 
# robotics where accurate spatial information is required