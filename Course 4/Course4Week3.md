# Week 3 

## Object Localization

Object localization is the process of getting an object detected in the given image and a box being drawn  aroung the detected object.

WE can treat the image with convolution and the output image we get output layer of 8 parameters in total if we have 3 classes to be classified from. which is in the case of detection of pedestrian, car, motorcycle in the given image. 

The 8 parameters will be of 1st parameter is if the object is present or not. 4 parameters will be of bx, by, bh,bw. These 4 varibles will respectively specify the centre of trhe box to be drawn as the ratio of width and height of the image and bh and bw is breadth and length of the box being drawn around the image. 

And the other 3 parameters will be of softmax between 3 different classes of pedestrian, car, motorcycle. 

The output y will be of the shape, y = [p, bx, by, bh, bw, C1 , C2, C3]

If there is no object in the image then the output will have p as 0 and all the other parameters will be dont care's i.e. y = [0,?, ?, ?, ?, ?, ?, ?]

## Landmark Detection

Landmark detection is when rather than identifying a box we need to identify number of points in the image. For e.g. we can locate points representing eyes, mouth and chin, for maybe face filters. We get 2 values for (x,y) coordinates, and an additional variable for whether the object is present of not. 

y = [p , lx1 , ly1 , lx2 , ly2 , ..... , lx_n , ly_n]


## Sliding Windows

We train a convolutional network, trained on image of cars which are zoomed and are of high quality. Giving Binar Classification on if the image is present of not. In this a filter of varying sizes, slides through the window giving multiple windows of all the possible boxes that can be created from the image. 

But this comes with a huge side-effect of massive computational cost. The computational Cost for this particular task is huge as all the sliding windows need to go through convolution network separately. 

We normally convert the image into 1x1xn to give a fully connected layer but for sliding window we get output as a matrix itself. For e.g. if there are 4 possible sliding windows, from all the corners then we can get 2x2xn where each box will contain information about its own sliding window. 

With this approach we are computing results for all the sliding window simulatneuosly rather than separately putting every window separately into convolutional network.

For example if we are given image of 28 x 28 x 3 to 8 x 8 x 4 due to 8 different sliding windows. 
There 4 parameters associated with all the sliding window grids will show, bx,by,bh,bw which will give boxes.

Nwo the issue with sliding window categorization is that the box created is broad and not actually tailored to the particular object. Maybe the car has a bit more width than height, and not perfect squares.

## YOLO Algorithm Intuition

We will divide the photo into grids, for e.g. let is divide the photo into 9 parts of 3 x 3. Then we will get output of matrix 3 x 3 x 8. Where each grid will have 8 parameters which show y = [p, bx, by, bh, bw, C1 , C2, C3]. By considering top left corner as (0,0) and bottom right corner as (1,1).

## Intersection Over Union

Most of the time there as train boxes which are formed for the object by getting the Intesection, Union Ratio with the actual box of the object. This gives us the ratio which gives the amount of overlapping. This gives a parameter to guess how good the generated box is. 

We normally give a box a pass, if the IoU is >= 0.5 

## Non - Max Supression

We normally dont have only 3 x 3 grid for the image but 19 x 19 or close to it. Due to which multiple boxes get created for the same object having different IoU's.

Algorithm for Non-Max Supression is as follows. 

Firstly we eliminate all the boxes with Pc < 0. This is the probabilty if the object is present or not. This gets rid of all the boxes which are not useful

Secondly we discard the IoU <= 0.5. from the remaining boxes.

As the name suggests the boxes which are NOT MAXIMUM in IoU or Pc are getting SUPRESSED.

## Anchor Boxes

We use anchor boxes in case of overlapping objects. Overlapping objects are generally a rare occurance but id there is a girl standing in front of a car then their will be two overlapping boxes for borth girl and car separately which will create ambiguity for the centre of the boxes. 

To get rid of this, We use anchor boxes. i.e. we create different box shapes for diffent objects. Like a bit elongated for a girl and a bit flattened for a car. 

Now instead of y being of dimension 8, it will be of dimension 16. This is because it carries information of both anchor boxes. If n number of anchor boxes are used then the number of parameters in the output function will be n*16

The box which will be generated, will first hae a IoU for the two anchor boxes which have been created. The anchor box with the largest IoU will be the object for the box. This eliminated confusion which could've been caused because of overlapping boxes.

## Putting YOLO Algorithm together

Let us consider 2 classes, 1. Pedestrian and 2. Car. Then the otput y for all the grids will be:

y = [p, bx, by, bh, bw, C1 , C2, C3, p, bx, by, bh, bw, C1 , C2, C3]

Each grid will have 2 predicted bounding boxes. After that we will get rid of low probability predictions for e.g. Pc < 0.6. 

After that we will use Non- Max Supression to get the best Box. We need to run this seperately for different anchor boxes. 

The final output will be of Size (g x g x a x (5 + classes))

Where g is the number of grid we've divided the picture into, a is the number fo anchor boxes and classes is the number of classed through which we have to classify.

## Semantic Segmentation

Semantic Segmentation is the process of giving each pixel a classification and getting a colour map as an output image, where each colour gives different object. In contrast to object detection in which we get objects detected and a box drawn around them.

This happens because of per pixel classification. Each pixel gets classified into given amount of classes.

If there is a car on road in fron of the house, then in segmentation map. The car pixels gets labelled 1, the house pixels gets labelled 2 and road pixels gets labelled 3.

For Deep Semantic Segmentation, we first use normal convolution and convert this into a small height and width with huge number of channels. Then we convert this into the image size again by increasing its dimensions to the input image size.

We increase the size of the image by using Transpose Convolution. 

Let us for example we need to convert 2 x 2 into a 4 x 4 imaeg using 3 x 3 filter. We first take the first element of the 2 x 2 image then multiply that number with all the numbers in 3 x 3. We pad 4 x 4 and make them to 6 x 6. We copy the the 3 x 3 filter on the 6 x 6 the filter on top left and reject the values in padding. 

We then take a stride of maybe 2 horizontally, similarly we take 2nd element from 2 x 2, multiply it to all the elements to filter and paste it to 4 x 4, and if some elements overlap then we have to add them together.

Similar process is going to happen till every combination is satisfied, and we get final 4 x 4 filter.

**U-Net:** It is a network which uses normal convolutions to compress the image into high feature extracted layer, then we increase its size again to get new image. We use skip connection during increasing the size so as to save the high resolution of the original image which is present in previous steps. 

 

