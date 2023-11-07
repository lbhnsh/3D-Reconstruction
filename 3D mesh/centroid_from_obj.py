vertices = []  # List to store vertex coordinates

with open('/home/param/Downloads/0_mesh_table_0.994(1).obj', 'r') as obj_file:
    for line in obj_file:
        if line.startswith('v '):
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            vertices.append((x, y, z))
print(vertices[0])
print(vertices[1])
print(vertices[2])
print(vertices[2023])
print(vertices[3000])

total_x = 0
total_y = 0
total_z = 0

for vertex in vertices:
    total_x += vertex[0]
    total_y += vertex[1]
    total_z += vertex[2]

num_vertices = len(vertices)
print(num_vertices)
centroid_x = total_x / num_vertices
centroid_y = total_y / num_vertices
centroid_z = total_z / num_vertices

centroid = (centroid_x, centroid_y, centroid_z)
print(centroid)

