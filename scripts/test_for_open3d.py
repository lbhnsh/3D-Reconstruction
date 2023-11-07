
import open3d as o3d
import numpy as np
# import sys
# sys.path.append('..')
# pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
# print(pcd)

print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# downpcd = pcd.voxel_down_sample(voxel_size=0.01)
# o3d.visualization.draw_geometries([downpcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

alpha = 0.03
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg

import matplotlib.pyplot as plt
import os
import sys

# sys.path.append('..')
import open3d_tutorial as o3dtut
o3dtut.interactive = not "CI" in os.environ

#RGBD IMAGES
# Redwood dataset

print("Read Redwood dataset")
# color_raw = o3d.io.read_image("/home/labhansh/Open3D/midas_test_0.jpg")
color_raw = o3d.io.read_image("/home/labhansh/test_data/RGBD/color/00000.jpg")
# color_raw = o3d.io.read_image("/home/labhansh/Open3D/copy_of_Juneau.png")
# depth_raw = o3d.io.read_image("/home/labhansh/Open3D/midas_test_0-dpt_beit_large_512.png")
depth_raw = o3d.io.read_image("/home/labhansh/test_data/RGBD/depth/00000.png")

rgbd_image= o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw,depth_raw)
print("HERE", rgbd_image)

plt.subplot(1,2,1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1,2,2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()

pcd=o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Flipping, else point cloud would be upside down
# pcd.transform([1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1])
o3d.visualization.draw_geometries([pcd], zoom=0.5)

'''

'''
print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
print("YO",pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


'''