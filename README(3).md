
# 3D-Reconstruction from Single RGB Image

Indentification of objects from a single RGB input image and generating separate masked images using YOLOv8 and YOLO-SAM architectures, generating 3-dimensional mesh using MeshRCNN for every masked image and concatenating them based on their x, y and relative z space co-ordinates to generate a 3D-reconstruction of the scene.



![s](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/outputs/test.png?raw=true)


![s](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/outputs/Screenshot%20from%202023-11-07%2023-42-56.png?raw=true)











## About the Project


## Table of Contents
* [Tech Stack](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#tech-stack)
* [Installations and preprocessing](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#installations-and-preprocessing)
* [Process Flow](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#process-flow)
* File Structure
* Execution
* Results
* Future Work
* Contributors
* [Acknowledgements and Resources](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README(2).md#acknowledgements-and-resources)


## Tech Stack

[Open3d (Build from Source)](http://www.open3d.org/docs/release/compilation.html)

[MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)

[Pix3D dataset](http://pix3d.csail.mit.edu/)







## Installations and Preprocessing

Tested on [Ubuntu 20.04](https://ubuntu.com/download/desktop)


[MeshRCNN](https://github.com/facebookresearch/meshrcnn)



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


## Results

![](https://github.com/lbhnsh/3D-Reconstruction/blob/main/final_results/input1.jpg?raw=true)
![model_output1.gif](https://github.com/lbhnsh/3D-Reconstruction/blob/main/final_results/model_output1.gif?raw=true)


## Future Work
## Acknowledgements and Resources

[Pixel2Mesh](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf)

[Depth-aware Pixel2Mesh](http://cs231n.stanford.edu/reports/2022/pdfs/167.pdf) 

[Holistic 3D scene Understanding](https://arxiv.org/pdf/2103.06422v3.pdf)

[YOLOv8 with YOLO-SAM](https://blog.roboflow.com/how-to-use-yolov8-with-sam/)


