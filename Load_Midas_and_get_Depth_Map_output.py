import open3d as o3d
import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# def save(filepath, fig):
#     '''Save the current image with no whitespace
#     Example filepath: "myfig.png" or r"C:\myfig.pdf" 
#     '''
#     if not fig:
#         fig = plt.gcf()

#     plt.subplots_adjust(0,0,1,1,0,0)
#     for ax in fig.axes:
#         ax.axis('off')
#         ax.margins(0,0)
#         ax.xaxis.set_major_locator(plt.NullLocator())
#         ax.yaxis.set_major_locator(plt.NullLocator())
#     fig.savefig(filepath, pad_inches = 0, bbox_inches='tight')


filename = ("/home/mrudul/Example/ROOM_RGB.jpg")


# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()


output = prediction.cpu().numpy()
# plt.imshow(output)
# plt.show()
# print("Prediction",type(prediction))
# print("Output",type(output))
# new_filename = "ROOM_DEPTH_MAP.png"
# cv2.imwrite(new_filename,output)

# panoptic_segmented_path="/home/mrudul/Example/IDK.jpg"
# plt.savefig(panoptic_segmented_path,bbox_inches='tight',pad_inches=0)
# plt.show()
# H,W = output.shape
print(output.shape)
print(img.shape)
# RGB = cv2.imread("/home/mrudul/Example/COMEON.jpg")
# DEPTH = cv2.imread("")
# GRAY = cv2.cvtColor(RGB,cv2.COLOR_BGR2GRAY)
# cv2.imwrite("/home/mrudul/Example/ROOM_DEPTH_GRAY.png",GRAY)

# CHECK = cv2.imread("/home/mrudul/Example/GRAY.png")
# print("Please: ",CHECK.shape)

# image = Image.open("/home/mrudul/Example/Dog_Real.png")
# print(image.shape)
# new_image = image.resize((W,H,0))
# new_image.save("/home/mrudul/Example/Genuine.png")
# print("Output:",output.shape)
# DEPTH = cv2.imread("/home/mrudul/Example/ROOM_DEPTH_GRAY.png")
# print("Depth Shape:",DEPTH.shape)
# RGB = cv2.imread("/home/mrudul/Example/ROOM_RGB.jpg")
# # print("RGB Shape:",RGB.shape)

# H,W,_ = RGB.shape
# image = Image.open("/home/mrudul/Example/ROOM_DEPTH_GRAY.png")

# new_image = image.resize((W,H))
# new_image.save("/home/mrudul/Example/REAL_GRAY_ROOM.png")

# rgb_image = cv2.imread("/home/mrudul/Example/dog.jpg")
# depth_image = cv2.imread("/home/mrudul/Example/GRAY.png")

# cv2.imshow("/home/mrudul/Example/dog.jpg",rgb_image)
# cv2.waitKey(10)

# RGB = Image.open("/home/mrudul/Example/dog.jpg")
# RGB.show()

# rgb_image = o3d.io.read_image("/home/mrudul/Example/ROOM_RGB.jpg")
# depth_image = o3d.io.read_image("/home/mrudul/Example/REAL_GRAY_ROOM.png")


# rgb_image = cv2.imread("/home/mrudul/Example/ROOM_RGB.jpg")
# cv2.imwrite("/home/mrudul/Example/ROOM_DEPTH_GRAY.png",GRAY)
# depth_image = cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
# print("RGB:",rgb_image.shape)
# print("DEPTH: ",depth_image.shape)
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,depth_image)


# plt.subplot(1,2,1)
# plt.title('Colour Image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1,2,2)
# plt.title('Depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()



# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image,
#     o3d.camera.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# # Flip it, otherwise the pointcloud will be upside down
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([pcd])


# downpcd = pcd.voxel_down_sample(voxel_size=1)
# o3d.visualization.draw_geometries([downpcd])
# ply_point_cloud = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])