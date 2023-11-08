# 3D-Reconstruction from Single RGB Image

## Project:
The model takes a single RGB image as input and attempts at creating a 3D mesh of the scene visible in the image by the methods of panoptic segmentation, masking, Mesh R CNN and then concatenation of alignment aware meshes to present the output.

### Input Image:
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result2/model_input2.jpg?raw=true)


### Output:
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result2/model_output2.gif?raw=true)



## Table of Contents
* [Process Flow](https://github.com/lbhnsh/3D-Reconstruction/tree/Final#process-flow)
* [File Structure](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#file-structure)
* [Installations and preprocessing](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#installations-and-preprocessing)
* [Execution](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#execution)
* [Results](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#results)
* [Tech Stack](https://github.com/lbhnsh/3D-Reconstruction/tree/Final#tech-stack)
* [Future Prospects](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#future-prospects)
* [Contributors](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#contributors)
* [Acknowledgements and Resources](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#acknowledgements-and-resources)


## About the Project:

Our Approach consists of first performing Panoptic Segmentation on the given image. This step associates distinct objects present in the scene with different hues. This association of objects with defined hues is then used to create masks of those object from the input RGB Image.

We create masks in order to aid the Mask-RCNN modality which is responsible to create masks for the objects present and then Mesh-RCNN creates mesh of the important objects present in the image. 

After the meshes are produced, they are then concatenated together in order to reconstruct the complete 3D Scene. Concatenation should result in the meshes being perfectly aligned with each other and with the camera as present in the input RGB image

### Process Flow
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/workflow.jpg?raw=true)
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/Screenshot%20from%202023-11-08%2002-50-38.png?raw=true)


1. Model takes RGB image as input in .jpg or .png file format 

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/input1.jpg?raw=true)

2. Panoptic segmentation: 

In semantic segmentation, all images of a pixel belong to a specific class. In instance segmentation, each object gets a unique identifier and appears as an extension of semantic segmentation. Panoptic Segmentation combines the merits of both approaches and distinguishes different objects to identify separate instances of each kind of object in the input image.

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/segmented_image.png?raw=true)


2.I. Generation of separate mask for every instance. Save only instances of those classes on which model is trained

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/segmented_rgb_images/segment_rgb_121_2.png?raw=true) 
      
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/segmented_rgb_images/segment_rgb_121_4.png?raw=true)
   
2.II. Make mesh for every mask using MeshRCNN
      
MeshRCNN predicts and aligns 3D-voxelised models using graphical convolutional network. *Inference can be run on Colab T4 GPU*
      
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/sofa.gif?raw=true)

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/table.gif?raw=true)
   
2.III. Merge meshes into one object file according to alignment of x-, y- and z- axes in RGBD plane
   
   ![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/model_output1.gif?raw=true)

## File Structure
```
📦3D-Reconstruction 
 ┣ 📂assets                            # Contains gifs, objs and images of the results 
 ┣ 📂scripts                           # C++ program used to run the drone
 ┃ ┣ segment_and_mask.py               # Used to create and save masks of objects from input image
 ┃ ┣ inference.ipynb                   # Run this notebook to get results
 ┣ 📜README.md
 ┣ 📜demo_video.gif                    # Demo Video
 ┣ 📜project_report.docx               # Project Report
 ┗ 📜requirements.txt                  # Requirements
``` 

## Installations and Preprocessing

Project was tested on Ubuntu 22.04 and T4 GPU offered by Google Colab

```pip install requirements.txt```

Rest all dependencies will be taken care of by the scripts

## Executing the Demo / Inference

Cloning into device 

```git clone https://github.com/lbhsnh/3D-Reconstruction.git```

```cd 3D-Reconstruction```

**Create a virtual env for the project**

```pip install requirements.txt```

```cd scripts```

```python3 segment_and_mask.py```

```then run the colab file inference.ipynb```


* To view .obj file
  
You can use Open3d to view the saved mesh or use :

[Online 3D Viewer](https://3dviewer.net/)


## Tech Stack

* Open3D

* Pytorch3D
  
* Detectron2
  
* Ultralytics


## Future Prospects

* Till now we’re able to create a combined mesh which aligns with the image. In future we aim at reconstruction of the wall and floor in order to create the entire scene present.

* The model is restricted only to a defined number of interior objects such as bed, couch, chair because of the limited number of classes present in the Pix3D dataset. We aim at improving the dataset by either finding a more diverse dataset or adding additional categories to the existing dataset.

* Due to GPU constraints we were unable to train the model to get an improved output. Therefore we plan to train on our new modified and diverse dataset in order to improve the diversity as well as the quality of mesh being produced.

## Mentor
* [Soham Mulye](https://github.com/Shazam213)
## Contributors

* [Labhansh Naik](https://github.com/lbhnsh)
* [Param Parekh](https://github.com/Param1304)
* [Mrudul Pawar](https://github.com/Mr-MVP)

## Acknowledgements and Resources

[Open 3D library documentation](http://www.open3d.org/docs/release/)

[Pixel2Mesh Paper by Nanyang Wang et al](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf)

For Image Segmentation Methods
* https://huggingface.co/shi-labs/oneformer_coco_swin_large 
* https://huggingface.co/docs/transformers/main/en/model_doc/maskformer 
* https://huggingface.co/blog/mask2former 

[Mesh R CNN by Justin Johnson et al](https://arxiv.org/pdf/1906.02739.pdf)


[Detectron2](https://github.com/facebookresearch/detectron2)
