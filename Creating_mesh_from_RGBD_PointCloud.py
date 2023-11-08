import open3d as o3d
import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# print("Testing mesh in Open3D...")
# armadillo_mesh = o3d.data.ArmadilloMesh()
# mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])

# mesh = o3dtut.get_bunny_mesh()
# pcd = mesh.sample_points_poisson_disk(750)
rgb_image = o3d.io.read_image("/home/mrudul/Example/00000.jpg")
depth_image = o3d.io.read_image("/home/mrudul/Example/00000_MIDAS_GRAY_PNG.png")

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,depth_image)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])
alpha = 0.03
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])


# knot_mesh = o3d.data.KnotMesh()
# mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
