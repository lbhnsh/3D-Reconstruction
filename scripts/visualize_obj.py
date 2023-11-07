# MODIFY line 32 by replacing it with your desired path 

import open3d as o3d

# Load your .obj mesh files
'''
mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
mesh3 = o3d.io.read_triangle_mesh(mesh3_path)
.
.
.
meshn = o3d.io.read_triangle_mesh(meshn_path)
'''
mesh1 = o3d.io.read_triangle_mesh("/home/labhansh/Downloads/new_sofa.obj")
mesh2 = o3d.io.read_triangle_mesh("/home/labhansh/Downloads/supposed_plant.obj")
mesh3 = o3d.io.read_triangle_mesh("/home/labhansh/Downloads/new_chair.obj")
# mesh4 = o3d.io.read_triangle_mesh("/home/labhansh/Downloads/wall.obj")
# mesh5 = o3d.io.read_triangle_mesh("/home/labhansh/Downloads/floor.obj")
mesh6 = o3d.io.read_triangle_mesh("/home/labhansh/Downloads/nayafloor.obj")


# Merge the meshes
combined_mesh = mesh1  + mesh2 + mesh3 + mesh6 #+ mesh4 + mesh5
# You can use the "+" operator for mesh concatenation

# Visualize the merged mesh
o3d.visualization.draw_geometries([combined_mesh])
# Save the merged mesh

combined_mesh_path=input("Enter complete path in the format (/your/path/your_mesh.obj)")
# REFERENCE: o3d.io.write_triangle_mesh("/home/labhansh/Downloads/merged_mesh.obj", combined_mesh)
o3d.io.write_triangle_mesh(combined_mesh_path, combined_mesh)
