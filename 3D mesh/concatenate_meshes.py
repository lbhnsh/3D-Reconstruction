mesh1 = o3d.io.read_triangle_mesh("mesh1.obj")
mesh2 = o3d.io.read_triangle_mesh("mesh2.obj")
mesh3 = o3d.io.read_triangle_mesh("mesh3.obj")
combined_mesh = mesh1 + mesh2 + mesh3 
o3d.visualization.draw_geometries([combined_mesh])
