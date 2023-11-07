
# 3D-Reconstruction from Single RGB Image

Indentification of objects from a single RGB input image and generating separate masked images using YOLOv8 and YOLO-SAM architectures, generating 3-dimensional mesh using MeshRCNN for every masked image and concatenating them based on their x, y and relative z space co-ordinates to generate a 3D-reconstruction of the scene.



![s](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/outputs/test.png?raw=true)


![s](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/outputs/Screenshot%20from%202023-11-07%2023-42-56.png?raw=true)











## About the Project

By executing this project, we aim at demonstrating the latest technologies in computer vision to greconstruct a 2-Dimensional image to 3-Dimensional object file. Numerous approach can be followed to achieve that, some of which utilized in this execution have been documented below:

* From Depth map
   1. Depth map generation from RGB image using MiDaS model
   ![](https://github.com/lbhnsh/3D-Reconstruction/blob/main/final_results/midas_depthmap.png?raw=True)
   2. Point cloud generation from RGB image and depth map using Open3D
   ![](https://github.com/lbhnsh/3D-Reconstruction/blob/main/pointcloud/pointcloud.gif?raw=true)
   3. Mesh generation using Pixel2Mesh 
   However mesh generation by Pixel2Mesh is below par due to improper training over older and not updated weights. Also it has angle of vision limitations while working with single image input.

* Using YOLO
   1. Object detection using YOLOv8 and simultaneously separated mask generation using YOLO-SAM 
   2. Mesh generation for each masked object using MeshRCNN
   3. Concatenation of meshes into one object file
   

* Using Panoptic segmentation
   1. Generation of separate mask for every instance. Save only instances of those classes on which model is trained
   2. Make mesh for every mask using MeshRCNN
   3. Merge meshes into one object file according to alignment of x-, y- and z- axes in RGBD plane
   ![](https://github.com/lbhnsh/3D-Reconstruction/blob/main/model_output2.gif?raw=true)
   


## Table of Contents
* [Tech Stack](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#tech-stack)
* [Installations and preprocessing](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#installations-and-preprocessing)
* [Process Flow](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#process-flow)
* [File Structure](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(3).md#file-structure)
* [Execution](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(3).md#execution)
* [Results](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(3).md#results)
* [Future Work](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(3).md#future-work)
* Contributors
* [Acknowledgements and Resources](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#acknowledgements-and-resources)


## Tech Stack

[Open3d (Build from Source)](http://www.open3d.org/docs/release/compilation.html)

[MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)

[Pix3D dataset](http://pix3d.csail.mit.edu/)

[ShapeNet rendered images](ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz)








## Installations and Preprocessing

Tested on [Ubuntu 20.04](https://ubuntu.com/download/desktop)


[MeshRCNN](https://github.com/facebookresearch/meshrcnn)


[numpy v1.21.6](https://numpy.org/)

[matplotlib v3.1.3](https://matplotlib.org/)

[Open3D v0.17.0](http://www.open3d.org/docs/release/getting_started.html)

[Ultralytics v8.0.200](https://docs.ultralytics.com/quickstart/)






## Process Flow

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/Screenshot%20from%202023-11-08%2002-50-38.png?raw=true)





## File Structure

* [instance_segmentation.py](https://github.com/lbhnsh/3D-Reconstruction/blob/main/scripts/instance_segmentation.py)
   * [segment_and_mask.py](https://github.com/lbhnsh/3D-Reconstruction/blob/main/demo/segment_and_mask.py)
   * [contour_based_segmentation.py](https://github.com/lbhnsh/3D-Reconstruction/blob/main/scripts/contour_based_segmentation.py)
* [inference.ipynb](https://github.com/lbhnsh/3D-Reconstruction/blob/main/demo/inference.ipynb)
* [visualize_obj.py](https://github.com/lbhnsh/3D-Reconstruction/blob/main/scripts/visualize_obj.py)
* [download_images.py](https://github.com/lbhnsh/3D-Reconstruction/blob/main/scripts/download_images.py)

## Execution

Cloning into device 

```git clone https://github.com/lbhsnh/3D-Reconstruction.git```

...
...

To view .obj file 

[Online 3D Viewer](https://3dviewer.net/)


## Results

![](https://github.com/lbhnsh/3D-Reconstruction/blob/main/final_results/input1.jpg?raw=true)
![model_output1.gif](https://github.com/lbhnsh/3D-Reconstruction/blob/main/final_results/model_output1.gif?raw=true)


## Future Work

1. Find or develop a better dataset
2. Train on that dataset and study output
3. 
## Contributors

* Labhansh Naik
* [Param Parekh](https://github.com/Param1304)
* Mrudul Pawar

## Acknowledgements and Resources

[Pixel2Mesh](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf)

[Depth-aware Pixel2Mesh](http://cs231n.stanford.edu/reports/2022/pdfs/167.pdf) 

[Holistic 3D scene Understanding](https://arxiv.org/pdf/2103.06422v3.pdf)

[YOLOv8 with YOLO-SAM](https://blog.roboflow.com/how-to-use-yolov8-with-sam/)


