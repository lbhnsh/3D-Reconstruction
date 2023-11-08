import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# import midas_implement as mid

# Load RGB and depth images & create RGBD image
# rgb_image = o3d.io.read_image(mid.input_path)
# depth_image = o3d.io.read_image(mid.output_path)
redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
rgb_image = o3d.io.read_image(redwood_rgbd.color_paths[0])
depth_image = o3d.io.read_image(redwood_rgbd.depth_paths[0])
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image)


# Showing Images
plt.subplot(1, 2, 1)
plt.title('Color image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()

# Creating Point Cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

# # Visualize the mesh
# o3d.visualization.draw_geometries([mesh])
