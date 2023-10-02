# Week 2

## Case Studies and their importance.

Case Studies are experiments and results published by researchers all around the world. rather than doing trial and error on everything ourselves we can study case studies of what other researchers did for the sam eproblem or the problem which resembled it. 

Major Network Architecture are LeNet , AlexNet, VGG, ResNet and Inception

## Types of Networks

1) AlexNet: AlexNet consists of 5 CONV layers, 3 MAX POOL layers and 3 F.C. Layers. In the order 

CONV-POOL-CONV-POOL-CONV-CONV-CONV-POOL-FC-FC-FC-Output

This network converts a 224 x 224 x 3 image, into an output Layer of 1000 options of Softmax. This network in total has 60M parameters.

2) VGG-16: It has in total 16 layers in the order

CONV[64] - POOL - CONV[128] - POOL - CONV[256] - POOL - CONV[512] - POOL - CONV[512] - POOL - FC - FC - Softmax


This is preffered as it has a straight forward non complex hyperparameter pattern in different layers, which is easier to follow. This Network in total has 138M parameters.

## Residual Networks

Residual Networks are networks which use skip connections between layers. This means that they concatenate activation output of some prior layer to the linear output of the current layer and then perform activation on it. 

We can look at this with an example

a[l] --> Linear --> ReLU --> a[l+1] --> Linear -->ReLU --> a[l+2]

In this we use the skip connection to join the value of a[l] in the step of Linear and ReLU between a[l+1] and a[l+2]. 

This results in the equation:

a[l+2] = g(z[l+2] + a[l])

In normal Plain Convolution Network, the training error decreases, reaches a minimum and then increases again. But in Residual Networks this doesn't happen and the network correctly keeps on decreasing the training error even if the number of layers keep on increasing. 

Residual networks are much more effective as they can learn identity function much more easily. 

If we consider the equation above.

a[l+2] = g((W[l+2]*a[l+1] + b[l+2]) + a[l])

We can see that even if due to Regularization or Vanishing Gradient, the Value fo W[l+2], begin to reduce The equation will just become a[l+2] = g(a[l]), which gives the original layers a much more importance in the later layers which increases the learning and eliminates vanushing gradient problem as well.

Wee might need to multiply a[l] with some weight matrix in order to make its size equal to a[l+2]

## 1x1 Convolutions

We mostly use filters of some size, but if we use a 1x1xn_c_prev filter on the image n x n x n_c_prev. It will give n x n image. It is as if for each pixel grid, the summation and activation of all the channels takes place to give teh new image. 

1x1 convolutions are mostly used to change the number of channels without affecting the size of the image being used. 

We can consider example of 6 x 6 x 128 and convolve it with n filters of size 1 x 1 x 128. It will give out 6 x 6 x number_of_filters.

We use 1x1 to reduce the number of parameters by them acting as a bottleneck.

For e.g. if we convert 28 x 28 x 192 we get 28 x 28 x 32 using 32 filters of size 5 x 5 x 192 then, the total number of parameter will be: 28\*28\*32\*5\*5\*192 = 120M

Now if we use a bottleneck 1 x 1 conv in between to convert the image first to 28 x 28 x 16 then to its final form the total number of parameter drops down to 12.4M which is a huge improvement.

## Inception Network

Inception Network is used when we Convolve the same image matrix using different sized filters, dividing the number of channels between them and adding all of them to get the new image size of same height, width and channels.

If we consider image matrix of 28 x 28 x 192 then use 1 x 1 filter to create 28 x 28 x 64 and then 3 x 3 to create 28 x 28 x 128, then 5 x 5 to create 28 x 28 x 32 and MAX pooling to create 28 x 28 x 32, while everything having Same Padding. to preserve the size of 28 x 28

For 3 x 3 and 5 x 5 we first use 1 x 1 to reduce their channels to 96 then increase their channels back to 128 using 3 x 3 conv and similar thing happens to 5 x 5.

After this all the image matrices from different conv filters and MAX pool is combined in one image matrix.

## MobileNets

MobileNet are special networks which are used for devices with low computational ability as not every machine can perform high computation on millions of parameters. 

In normal convolution example if we consider 6 x 6 x 3 image to be convolved with 5 filters of size 3 x 3 x 3 to create 4 x 4 x 5

The complutaional cost is 3 x 3 x 3 x 4 x 4 x 5 = 2160

Now Let us consider a computational cheaper alternative called DepthWise Separable Convolutions

**Depthwise Separable Convolutions**

1. Depthwise Convolution

We first perform a depthwise convolution and then perform a pointwise convoltion. 

Let us consider the example again having 6 x 6 x 3. which will have its 3 channels each of size 6 x 6 gets separately convolved with 3 x 3 filters. Which gives us 4 x 4 x 3. But instead of convolution over volume, convolution over area takes place in different channels separaely and gets combined. 

Computational Cost for will be: 3\*3\*4\*4\*3 = 432

2. Pointwise Convolution

Then the processed image of 4 x 4 x 3 will be convolved with 5 filters of 1 x 1 x 3. to give 4 x 4 x 5. 

The Computational Cost for this step will be: 1\*1\*3\*4\*4\*5 = 240

**Cost Summary**:

By normal Convolution we get 2160 

And with Depthwise Separable we get 432 + 240 = 672

The ratio that we will get is 672/2160 i.e. 0.31 i.e. the Depthwise separable convolution is 3 times more computational efficient than normal convolutions.

Normally this ratio is given by 1/n + 1/f^2 where n is number of channels in output and f is filter size and normally number of channels is high such as 512 which results in this new architecture being upto 10 times more cost efficient. 

## Different Versions of MobileNet

MobileNet v1: It uses depthwise separable convolutions 13 times.

MobileNet v2: It uses a different approach block which has expansion convolution , depthwise convolution and then pointwise convolution, the part prior to expansion convolution and after pointwise convolution is connected via a skip connection. 

For example if we consider n x n x 3 image then what expansion does is it increases the amount of channels which results in increased feature extracion and converts to n x n x 18, depthwise convolution takes place which creates a n x n x 18. Then using pointwise convolution with 3 filters of size 1 x 1 x 18 we get final nxnx3. The info prior to exapansion and after pointwise is connected through skip connection.

Actual Architecture on MobileNet v2 consists of:

CONV2D --> 17 x [Exp-Depth-Point Block] --> CONV2D[1x1] --> avgpool[7x7] --> CONV2D[1x1]

Another approach to tune computation with respect to the device being used is EfficientNet. We can tune n_h, n_w or r which represents channel which represents resolution. We can tune all these parameters and look at what paraemter we can compromise or enhance to give best output for the given machine.


## Transfer Learning and Data Augmentation

Generally for Computer Vision tasks we need a huge amount of Training examples, which is expensive and time consuming to get nbut also it is computational very costly to train all these images as well.

For this we can use open source codes, researchers or individuals who themselves have faced the problem and have trained thir model, and published them openly. 

We can use their already trained parameters for our problem. We just need to change a few neurons in the output layer or else we can freeze the entire model and change some of the last layers to tailor it according to our problem. Freezing the layers means back-propogation wouldn't affect it.

Data Augmentation is the process of creating new Image Data from existing examples. 

We can Mirror the image, rotate the image by a small angle, zoom and crop. We can change the pixel value as well. All of this gives us multiple examples using only one example. This is commonly used in Computer Vision tasks as training examples which are needed are very high.


 


