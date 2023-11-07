import torch 
import cv2
import urllib.request
import numpy
import matplotlib.pyplot as plt
import open3d as od
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

#load a model
model_name = "DPT_Large"
model_type = "DPT zhybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
#load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
#load image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#apply transform
input_batch = transform(img)
#predict and resize to original resolution
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size = img.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()
    
output = prediction.cpu().numpy()
#show result
plt.imshew()