# Convolutional Neural Networks

## WEEK 1:

### COMPUTER VISION

Computer Vision Problems:
1. Image Classification
2. Object Detection
3. Neural Style Transfer

ASTERISK IS A SYMBOL FOR CONVOLUTION
BUT IN PYTHON ITS FOR MULTIPLICATION

### Edge Detection Example

#### Vertical edge detection

Filter or a Kernel (3*3)

1 0 -1
1 0 -1
1 0 -1 

### More Edge Detection

#### Horizontal edge detection

1   1  1
0   0  0
-1 -1 -1

#### Learning to detect edges

(Vertical edge) 
1 0 -1 		3	0	-3
2 0 -2		10	0	-10
1 0 -1		3	0	-3

Sobel filter	Scharr filter


### Padding

n*n image *  f*f filter -> (n-f+1)*(n-f+1) output

Problems:
1. Shrinking output
2. Throwing away info from edges

Solution:
Pad the image with extra 1*1 layer outside existing image
i.e a 6*6 image will be 8*8 after padding

p=padding=1 (in this case 1)


#### Valid and Same Convolutions

##### Valid = No Padding

##### Same = Pad so that output size is same as input image

**FOR SAME CONVOLUTIONS**
p= (f-1)/2


~f is generally odd

### Strided Convolutions

stride is the jump of a filter
if a stride=2 the filter moves after 2 points


AFTER STRIDED CONVOLUTION
n*n image, f*f filter
then
output:
|(n+2p-f)/s + 1| * |(n+2p-f)/s + 1|

where | x | is step


MATHEMATICALLY its called CROSS-CORRELATION
but we call it CONVOLUTION in DEEP LEARNING


### Convolutions Over Volume

We can Convolve with a 3D filter 

**IMAGE AND FILTER SHOULD HAVE SAME NO. OF CHANNELS**

After applying different filters the outputs can be stacked together as convolution volumes

### One layer of a Convolutional Network

If Layer l is a convolution layer:

f[l]= filter size
p[l]= padding
s[l]=stride

Input: nh[l-1]*nw[l-1]*nc[l-1]
Output: nh[l-1]*nw[l-1]*nc[l-1]

n[l]=|(n[l-1]+2p[l]-f[l])/s[l] + 1|   |step|

each filter size:
f[l]*f[l]*nc[l-1]

Activations dimension:
a[l] = nh[l]*nw[l]*nc[l]
A[l] = m*nh[l]*nw[l]*nc[l]

Weights dimension:
f[l]*f[l]*nc[l-1]*nc[l]

Bias dimension: 
nc[l]


### Simple Convolutional Network Example

Example ConvNet


Types of Layer:
1. Convolution (CONV)
2. Pooling (POOL)
3. Fully Connected (FC)



















## Week 2:

## Why look at Case studies?

Classic Networks:
1. LeNet-5
2. AlexNet
3. VGG

ResNet (152 layers)
Inception


### Classic Networks:
LeNet-5
5*5 filter with stride 1 -> avg pool (s=2,f=2) -> 5*5 filter with stride 1 -> avg pool (s=2,f=2) -> Fully Connected (120) -> Fully Connected (84) -> Softmax

AlexNet

227*227*3 
11*11, s=4 -> Max Pool (s=2,f=3) -> 5*5 same -> Max Pool (s=2,f=3) -> 3*3 same -> 3*3 -> 3*3 -> Max Pool (s=2,f=3) -> 9216 -> (Fully connected) 4096->(Fully connected)  4096 -> softmax (100)

VGG-16
CONV = 3*3. s=1, same
MAX-POOL = 2*2, s=2

[CONV 64] 2 times -> POOL -> [CONV 128] 2 times -> POOL -> [CONV 256]*3 -> POOL ->,,,,,




### ResNets

Residual Networks:
Residual Blocks:

Skip connection (short cut)

purpose is to move much deeper into the network

### Why ResNets Work?

Addition of Residual block doesnt hurt the learning process and performance

in ResNets **SAME CONVOLUTIONS** are used so identification is easy.

### Networks in Networks and 1x1 Convolutions

This is used to shrink the n_C 
Pooling can only shrink n_H, n_W

### Inception Network Motivation

The n_C in inception output layer are concatenated ( some of 1*1, 3*3, 5*5, Maxpool convolutions)


Bottleneck layer: compressed

### Inception Module:
Concatenation of different convolutions

### MobileNet

Low Computational cost at deployment
Computational Cost = # filter params * # filter positions * # of filters

Depthwise Seperable Convolution

1. Depthwise convolution: Filter will have size onlt f* f and # filters will be n_C

2. Pointwise convolution: Output of Depthwise Convolution  * (1*1*n_C) = n_out*n_out*n_C' 
(n_C' = # filters)

Benefit of Depthwise Seperable over Normal Convolution is less Computational cost

### MobileNet Architecture

MOBILENETv1
(FC= fully connected)
Depthwise Convolution 13 times -> Pooling -> FC -> SOFTMAX

MOBILENETv2
(FC= fully connected)
inPUT (Residual Connection )Depthwise Convolution 17 times -> Pooling -> FC -> SOFTMAX

### EfficientNet

To know which factors better: resolution, width, depth of layers


### Data Augmentation

Techniques:
1. mirroring
2. random cropping
3. rotation
4. shearing
5. local warping
6. **color shifting (changing r,g,b values)**

Advanced:
PCA color augmentation 
Principles Component Analysis



## Week 3:
Object Detection

### Object Localization
What are localization and detection?
Figuring out where object is: localization
Detect which object: classification
Multiple objects: detection

Defining Target Label 'y'

y=[	pc
	bx
	by
	bh
	bw
	c1
	c2
	c3 	]

pc= is there an object?

if pc=0 rest all values are '?' don't cares

### Landmark Detection

### Object Detection
(SLIDING WINDOW ALGORITHM)

1. Apply ConvNet for the small Window you have chosen and slide it till the image ends. (iterate through every region)
2. Repeat but increase window size
3. Repeat but increase window size again

Disadvantage: High Computational Cost


### Convolutional Implementation of Sliding Windows

Turning FC layer into COnvolutional layer

Convolution Implementation of Sliding Windows

To reduce computation:
Share parameters:
adding extra (padding like layers) 


### Bounding Box Predictions:

YOLO (You Only Look Once) Algorithm:
Distribute image into grids and apply the Convolutional implementation to each grid cell, but it runs only once sharing the parameters and less computational cost

Find the coordinate of the centre of the object detected and find the distances of the object


### Intersection Over Union (IoU)

IoU = size of intersection/ size of union

Correct if IoU>= 0.5

measure of overlap between two bounding boxes


### Non-Max Suppresion Example
multiple grids may feel like the car has its midpoint in that grid cell
that will lead to have multiple detection boxes
non-max supression will eradicate the problem 

Only the rectangle with highest pc will be highlighted and rest will be suppressed

Discard all boxes with pc<=0.6
	While there are any remaining boxes:
		1. pick the box with largest pc (output that as prediction)
		2. Discard any reamining box with IoU>=0.5 with box output in the previous step
	
### Anchor Boxes:
How to detect multiple overlapping objects?

Predefine shapes of Anchor boxes ( shape of anchor boxes for a type of object )

y will have 16 values now instead of 8
first anchor box1 then anchor box2

Output: n*n* 16

(grid cell, anchor box)

y=[	pc		<- Anchor box 1 (start)
	bx
	by
	bh
	bw
	c1
	c2
	c3 		<- Anchor box 1 (end)
	pc		<- Anchor box 2 (start)
	bx
	by
	bh
	bw
	c1
	c2
	c3 	]	<- Anchor box 2 (end)

### YOLO Algorithm:

y is n*n*#anchors*(5+#classes)

#### Outputting the non-max suppressed outputs:
1. For each grid cell, get 2 predicted bounding boxes
2. Get rid of low probab predictions
3. For each class use non-max suppression to generate final predictions


### Semantic Segmentation with U-Net

Per-pixel class labels

Deep learning for Semantic Segmentation
(key for Semantic Segmentation is it needs to upscale itself)

### Transpose Convolutions

Used to upscale

Place filter on the output instead of input
(refer video)
[https://www.coursera.org/learn/convolutional-neural-networks/lecture/kyoqR/transpose-convolutions] (Transpose Convolutions)

### U-Net Architecture Intuition

First half Normal Convolutions
Later half Transpose Convolutions

Uses Skip connections: matching dimension layers 

### U-Net Architecture

Conv,RELU ->  Max Pool
 (Skip connections in between)-> TransConv -> 1*1 Convolution
 
 
 
## Week 4:
 
### What is Face Recognition?

 Verification:
 1. input image,name/id
 2. output if image is of that person
 
 Recognition:
 1. has a database of k person
 2. get an input image
 3. output id if the image is any of the k persons (or not recognized)
 
### One Shot Learning

(only 1 example available)

Learning a similarity function (d)
find degree of difference between images (one with least degree of difference is the output)

### Siamense network

training two different (same parameters) networks ( for different images) to find 'd'

### Triplet Loss

Learning Objective:
Look at Three Images at a time
A= Anchor (sample)
P= Positive (matching)
N= Negative



d(A,P)= ||f(A)-f(P)||^2 
d(A,N)= ||f(A)-f(N)||^2

d(A,P) -d(A,N) + alpha <=0


Loss function:
Given 3 images A,P,N:
L(A,P,N) = max(d(A,P) -d(A,N) + alpha ,0)

choose triplets that're hard to train on
when d(A,P) is almost equal to d(A,N)


### Face verification and Binary Classification

yhat​=σ(∑k=1,128 ​wk​∣f(x(i))k​−f(x(j))k​∣+b)



### Neural Style Transfer
#### What are deep ConvNets learning?
initial layers learn simpler features (like edges) 
and deeper layers see and learn larger patches of the image

#### Cost function:

J(G) = alpha*Jcontent(C,G) + beta*Jstyle(S,G)

G:= G - dG

#### Content Cost Function

Jcontent(C,G) = 1/2 ||a [l] (C) -a [l] (G)||^2

#### Style Cost Function

Gkk′[l] (G)​ =∑i=1nH​​∑j=1nW​​ ai,j,k [l] (G)​ai,j,k′[l] (G)​.


Jstyle[l] ​(S,G)=1/(2nH[l]​nW[l]​nC[l]​)^2  *  1​∑k​∑k′​(Gkk′[l] (S)​−Gkk′[l] (G)​)^2


#### 1D and 3D generalizations

