# 3D-Reconstruction from Single RGB Image

## Project:
The model takes a single RGB image as input and attempts at creating a 3D mesh of the scene visible in the image by the methods of panoptic segmentation, masking, Mesh R CNN and then concatenation of alignment aware meshes to present the output.

### Input Image: 
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result2/model_input2.jpg?raw=true)


### Output:
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result2/model_output2.gif?raw=true)



## Table of Contents
* [About the Project](https://github.com/lbhnsh/3D-Reconstruction/tree/Final#about-the-project)
* [Process Flow](https://github.com/lbhnsh/3D-Reconstruction/tree/Final#process-flow)
* [File Structure](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#file-structure)
* [Architecture and Dataset](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#architecture-and-dataset)
* [Installations and Execution](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#installations-and-execution)
* [Results](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/demo_video.gif)
* [Tech Stack](https://github.com/lbhnsh/3D-Reconstruction/tree/Final#tech-stack)
* [Future Prospects](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#future-prospects)
* [Contributors](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#contributors)
* [Acknowledgements and Resources](https://github.com/lbhnsh/3D-Reconstruction/blob/Final/README.md#acknowledgements-and-resources)


## About the Project:

Our Approach consists of first performing Panoptic Segmentation on the given image. This step associates distinct objects present in the scene with unique hues. This association of objects with unique hues is then used to create masks of those object from the input RGB Image.

We create masks in order to aid the Mask-RCNN modality which is responsible to create masks for the objects present and then Mesh-RCNN creates mesh of the important objects present in the image. 

After the meshes are produced, they are then concatenated together in order to reconstruct the complete 3D Scene. Concatenation should result in the meshes being perfectly aligned with each other and with the camera as present in the input RGB image

We've used ShapeNet Dataset which contains huge CAD amounts of model from diverse categories. This dataset is standard when it comes to ML model building for 3D applications. We've also evaluated our model on the challenging dataset of Pix3D. This dataset consists of real life images and models of objects which are aligned with the image provided making it a one of a kind dataset, as it helps yield reasonable output even when challenged with real-life images.

### Process Flow

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/workflow.jpg?raw=true)
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Param-Parekh/Screenshot%20from%202023-11-08%2002-50-38.png?raw=true)


1. **Input**: The complete model will have the input in the form of a single RGB image. The image file can be in .jpg or .png file format. 

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/input1.jpg?raw=true)


2. **Panoptic segmentation:** For the given image with our ML Model, panoptic segmentation will be applied on the given input image. As Panoptic Segmentation is the combination of instance segmentation as well as semantic segmentation, we get the regions of the objects present but as well as the distinct regions  different classification of objects present in the scene

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/segmented_image.png?raw=true)


3. **Generation of masks**: With the help of the regions obtained by the Panoptic Segmentation, we then move towards generating masks of the distinct object instances present in the image. We perform this step specially to aid the formation of better masks by the Mask RCNN which is the primary input for the Mesh Modality which creates mesh for individual objects. 

Generation of separate mask for every instance. Save only instances of those classes on which model is trained

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/segmented_rgb_images/segment_rgb_121_2.png?raw=true) 
      
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/segmented_rgb_images/segment_rgb_121_4.png?raw=true)
   
4. **Generation of individual mesh**: By the now obtained refined and accurate masks of the the objects, mesh are created singular objects by the mesh formation block applied in Mesh RCNN. A rough voxel grid is first formed for the image and which is then refined by Mesh Refinement, following a coarse-to-fine approach which creates an ideal mesh. 

   
*Inference can be run on Colab T4 GPU*
      
![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/sofa.gif?raw=true)

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/table.gif?raw=true)
   
5. **Concatenation of meshes**: We use the functions offered by the Open3D library in order to achieve the final mesh. The final mesh consists of all the previous individual meshes aligned with each other and with the camera as present in the input image.
   
   ![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/result1/model_output1.gif?raw=true)
   

## File Structure
```
📦3D-Reconstruction 
 ┣ 📂assets                            # Contains gifs, objs and images of the results 
 ┣ 📂scripts                           # Python programs to run 
 ┃ ┣ segment_and_mask.py               # Used to create and save masks of objects from input image
 ┃ ┣ inference.ipynb                   # Run this notebook to get results
 ┣ 📜README.md
 ┣ 📜demo_video.gif                    # Demo Video
 ┣ 📜project_report.docx               # Project Report
 ┗ 📜requirements.txt                  # Requirements
``` 
## Architecture and Dataset

The Mesh that gets generated from the masked image is done on the basis of the [Mesh RCNN](https://arxiv.org/pdf/1906.02739.pdf) architecture

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/MeshRCNNarch.png?raw=true)

It has been trained upon [Pix3D](http://pix3d.csail.mit.edu/) dataset

![](https://github.com/lbhnsh/3D-Reconstruction/blob/Labhansh-Naik/assets/pix3d.png?raw=true)


## Installations and Execution

Project was tested on Ubuntu 22.04 and T4 GPU offered by Google Colab


Cloning into device 

```git clone https://github.com/lbhsnh/3D-Reconstruction.git```

```cd 3D-Reconstruction```

**Create a virtual env for the project**

```pip install requirements.txt```

Rest all dependencies will be taken care of by the scripts

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
