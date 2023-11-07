import cv2
import open3d as o3d
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


#Load model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

#Load transforms to resize and normalize the image for large or small model

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# input_path="/home/labhansh/Open3D/MiDaS/input/cat_og.jpg"
input_path=input("Enter input image path: (/your/path/image.jpg): ")
rgb_image = o3d.io.read_image(input_path)
output_path=input("Enter output image path: (/your/path/image.png): ")
#Load image and apply transforms
img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

input_batch = transform(img).to('cpu')

#Predict and resize to original resolution

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
# output_path = "/home/labhansh/Open3D/MiDaS/depthmap_segments/cat_depth.png"  # Replace with your desired path and file format
cv2.imwrite(output_path, (output))

plt.imshow(output)
plt.show()

