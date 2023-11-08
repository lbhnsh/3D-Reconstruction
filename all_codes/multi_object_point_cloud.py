import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# from huggingface_semantic import *
# replace segments with images
n_segments=5 # import from huggingface code later
segmented_point_clouds = []

# WORK WITH DEPTH IMAGE, NOT INVERSE DEPTH IMAGE
# VISUALIZE THE POINT CLOUD

for segment_id in range(n_segments):
    rgb_image = o3d.io.read_image(f'/home/labhansh/Open3D/segmented_rgb_images/segment_rgb_{segment_id}.png') #"/home/labhansh/Open3D/MiDaS/input_segments/segment1.png"
    depth_image = o3d.io.read_image(f'/home/labhansh/Open3D/segmented_depth_images/segment_depth_{segment_id}.png') #"/home/labhansh/Open3D/MiDaS/input_segments/depth_segment1.png"
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    segmented_point_clouds.append(pcd)
    # o3d.visualization.draw_geometries([pcd])
    # alpha = 0.03
    # print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

combined_point_cloud = o3d.geometry.PointCloud()


for segmented_point_cloud in segmented_point_clouds:
    combined_point_cloud += segmented_point_cloud

downpcd = combined_point_cloud
downpcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([downpcd])
# o3d.io.write_point_cloud("/home/labhansh/Open3D/MiDaS/store_pointclouds/tennis_guys.pcd", combined_point_cloud)

alpha = 0.03
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downpcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])