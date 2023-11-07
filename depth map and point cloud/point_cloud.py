color_raw = o3d.io.read_image("./path_of_RGB")
depth_raw = o3d.io.read_image("./path_of_Depth")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)