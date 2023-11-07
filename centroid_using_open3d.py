import open3d as o3d
import numpy as np
mesh1 = o3d.io.read_triangle_mesh("/home/param/Downloads/0_mesh_table_0.994.obj")
vertices = np.asarray(mesh1.vertices)
centroid1 = np.mean(vertices, axis=0)
print("Centroid of the mesh:", centroid1)

mesh2 = o3d.io.read_triangle_mesh("/home/param/Downloads/0_mesh_sofa_1.000.obj")
vertices = np.asarray(mesh2.vertices)
centroid2 = np.mean(vertices, axis=0)
print("Centroid of the mesh:", centroid2)


