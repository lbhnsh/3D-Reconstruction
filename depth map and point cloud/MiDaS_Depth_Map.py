import cv2
import torch
import numpy as np
import urllib.request
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import open3d as od
import imageio.v3 as iio
# url, filename = ("https://github.com/pytorch/hub/blob/master/images/dog.jpg?raw=true", "dog.jpg")
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# url, filename = ("https://production-media.paperswithcode.com/datasets/38404c57-cd43-4911-a7c3-3af914111afb.png?raw=true", "cars.jpg")
# url, filename = ("https://production-media.paperswithcode.com/datasets/38404c57-cd43-4911-a7c3-3af914111afb.png?raw=true", "cars.jpg")
urllib.request.urlretrieve(url,filename)

#load MiDas Model for depth estimation
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
#Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
#Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Load image and apply transform
img = cv2.imread(filename)
# img = cv2.imread("/home/param/Documents/dog.jpg")
# img = cv2.imread("dog.jpg")
# img = cv2.imread("cars.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)
#predict and resize to original resolution
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size = img.shape[:2],
        mode="bicubic",
        align_corners = False,
    ).squeeze()
output = prediction.cpu().numpy()
cv2.imwrite("/home/param/Downloads/Figure_2.png",output)
# /home/param/Downloads
plt.imshow(output)
plt.show()



#saving output image to python file
# img = plt.imread("Figure_1.png")
# plt.imshow(img)
# plt.savefig(img)
# image = Image.new('RGBD',(100,100), output)
# image.save(output)

# depth_map = Image.open("/home/param/Documents/Figure_1.png")
# depth_data = np.asarray(depth_map, dtype=np.float32)
# depth_data = np.flip(depth_data, axis=0)
# #print(depth_data.shape)
# height, width, depth = depth_data.shape
# pcd = od.geometry.PointCloud()
# #define intrinsic parameters
# fx = 1000
# fy = 1080
# cx = width//2
# cy = height//2
# intrinsic_matrix = np.array([[fx, 0, cx],
#                              [0, fy, cy],
#                              [0, 0, 1]])
# intrinsic = od.camera.PinholeCameraIntrinsic()
# intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
#ntrinsic_matrix = intrinsic.intrinsic_matrix
# pcd = od.geometry.PointCloud(
#     depth_map,
#     depth=od.geometry.Image(depth_data),
#     intrinsic=intrinsic_matrix
# )
# od.visualization.draw_geometries([pcd])

#read depth image
# print("Depth map image (RGBD)")
# depth_image = iio.imread("/home/param/Documents/Figure_1.png")
# #print properties
# print(f"Image resolution: {depth_image.shape}")
# print(f"Data type: {depth_image.dtype}")
# print(f"Min value: {np.min(depth_image)}")
# print(f"Max value: {np.max(depth_image)}")

# #depth_image = np.array(depth_image)

# #read original image
# print("Original RGB image")
# orig_image = iio.imread("/home/param/Documents/dog.jpg")
# print(f"Image resolution: {orig_image.shape}")
# print(f"Data type: {orig_image.dtype}")
# print(f"Min value: {np.min(orig_image)}")
# print(f"Max value: {np.max(orig_image)}")
# plt.imshow(orig_image)
# plt.show()
# #depth image is of size 480 X 640 where each pixel is 8 bit integer that represents distance in millimeters
# #min value = nearest pixel
# #max value = farthest pixel

# #create rgbd image
# rgbd_image = od.geometry.RGBDImage.create_from_color_and_depth(orig_image, depth_image, depth_scale = 1000.0, depth_trunc = 3.0, convert_rgb_to_intensity=True)

# #Showing images
# plt.subplot(1,3,1)
# plt.title("original RGB image")
# plt.imshow(orig_image)
# plt.subplot(1,3,2)
# plt.title("Depth map image")
# plt.imshow(depth_image)
# plt.subplot(1,3,3)
# plt.title("RGBD image")
# plt.imshow(rgbd_image)
# plt.show()

# pcd = od.geometry.PointCloud()
# fx = 525.0  # Focal length in x direction
# fy = 525.0  # Focal length in y direction
# cx = depth_image.shape[0] / 2  # Principal point in x direction
# cy = depth_image.shape[1] / 2  # Principal point in y direction
# intrinsic = od.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)
# extrinsic = np.ndarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
# pcd = od.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic), 
# od.visualization.draw_geometries([pcd])



# pcd = od.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image,
#     od.camera.PinholeCameraIntrinsic(
#         od.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
#     )
# )
# od.visualization.draw_geometries([pcd])


