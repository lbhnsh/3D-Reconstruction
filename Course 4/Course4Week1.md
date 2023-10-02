# Week 1

## Why convolution is important and what it can achieve

Convolution Neural Networks are widely used in image classfication problem. Some common tasks for which CNNs can be used are:

**Image Classification:** To tell whether a image consists of a certain object or not. We can build a cat classifier, identifying if it is a cat or not. 

**Object Detection:** We can detect certain objects and draw boxes around them classifying where they are and what they are in the picture.

**Neural Style Transfer:** We can combine two art styles into one image. We can select a particular picture and apply style of another artwork on it. 

We use Convolutional Neural Networks due to the huge amount of parameters being generated while using Normal Deep Neural Networks for the same image. Imagine we have a clear imaeg of a cat having dimensions 1000x1000x3. Followed by another layer of just 1000 units. We can see the parameters for these two layers alone will come out to be: 1000*1000\*3\*1000 which is **3 Billion**

Such huge amount of parameters are not feasable for both computation cost and time taken. Thats why we use convolutional Neural Networks


## Basic Edge Detection

We can detect edge in a particular image using certain filters. These filters are mostly 3x3 which has 1 in the first column, 0 in the second column and -1 in the third column. These are used when their is an edge, i.e. when the image has a contrast between a dark colour and a bright colour or 2 different colours. When this filter gets convolved into the image. The new image formed will have its edges highlighted. Simlar thing can be done for horizontal and digaonal edges.

There are differnet type of filters such as Sober filter and Scharr Filter. There are cases when, we use parameters instead of using fized values of the filters which then learns during iterations. But it is observed that the parameters of the filters will take the values somewhat near to the filters which we will be using. 

## Padding

We can see that the image size gets shrinked, when convolution takes place on the image using a particular filter. Because of the equation:

n_new = n - f + 1

Because of this we use padding. Padding is when we put an additional boundary of 0 pixel value on the image which is to be treated, in order to get an image of the same size which conseves the initial size of the image. 
There are 2 types of convolutions, Valid and Same. 

In Valid convolution, no padding takes place. But in Same Convolution padding does take place in order to create the image after convolution to have same size. The equation for padding becomes, p = (f-1)/2

## Strided Convolution

Strides is the pixel with which the filter convolves on the initial image. Till now the filter moved pizel by pizel horizontally and vertically, thats why the stride is 1. We can change the stride to n, depending upon the use. 

The new size of the new image formed after the convolution is n_new = (n + 2p - f)/s + 1

## Convolution over volumes

Generally, the images which we use in day to day life in this age are not black and white but are RGB. Thats why the size of the image are n x n x 3. Where 3 is for three separate channels for Red, Green and Blue. Thats why we use the filter of f x f x 3, where f is the filter size. 

What we do is, we convolve the f x f x 3 on the n x n x 3, And get a single value by element-wise multiplication and adding it to get the new element for the grid in the filter. If there is one filter then we will get the image of size n_new x n_new. But if there are filter_number , of filters use then the image size we will get is n_new x n_new x filter_number. Now these filters which are being used are different and are responsible for extracting different information about the image. 


## Understanding convolutions for One Layer CNN

If we consider a image of 6 x 6 x 3 and let us consider 2 filters of size 3 x 3 x 3. For first filter we will apply convolution after that it will undergo an activation function like ReLU and then bias will be added to it giving us the final image of size 4x4. Similar thing will happen with the second filter which gives us another 4x4 image. In the end wegather these 2 separate 4x4 matrices and group them into one matrix of size 4x4x2. 

In summary if we consider,

f[l] => filter size

p[l] => padding

s[l] => strides

n_c[l] => fnumber of filters. 

The dimensions of each filter will be  f[l] x f[l] x n_c[l-1]. And number of filters will be n_c[l]

The activation layer will have dimensions a[l] = n_w[l] x n_h[l] x n_c[l].

Weights will be w[l] = f[l] x f[l] x n_c[l-1] x n_c[l]

In the activation the new n_w[l] will be defined as 

n_w[l] = (n_w[l-1] + 2p -f)/s + 1

n_h[l] = (n_h[l-1] + 2p -f)/s + 1


We can consider a convolution network which shrinks the 6 x 6 x 3 example into a 1 x 1 x 400 after multiple convolutions which shrinks the size of the image. This matrix is then projected onto a 400 neuron layer. Called fully connected layers and then it is processed further with some more dense connected layer giving our final output layer, which can be simple sigmoid or a softmax function.

## Different types of layers in Convolution Neural Networks

There are 3 typed of Layers which make CNNs. 

1. Convolutional Layers

2. Pooling Layers

3. Fully Connected Layers

We have looked in detail about Convolutional Layers till now lets look at Max Pooling Layers

**Pooling Layers**

There are two types of pooling layers:
1) Max Pooling Layers:

Max pooling layers are used to change the size of the image i.e. n_h, n_w while keeping the number of channels the same. This is used to increase the amount of features being extracted as well as decreasing the image size for further computation. 

If we consider the, max pooling filter to be of size 2 x 2 and stride 2 x 2. If we apply this on the image 4x4. It will select the 2x2 squares of all the corners and put the maximum values of those squares into a new 2x2 image. 

After pooling layer the new image dimensions are: n_h_new = (n_h - f)/s + 1, n_w_new = (n_w - f)/s + 1 and n_c_new = n.

2) Average Pooling Layers: Exact same thing happens for the average pooling lyaers as well, but the only difference is that instead of the maximum value, average value of the 2x2 corner matrices of the image 4x4 will be taken and put into the new 2x2 image.

## LeNet 5

It is one of the first Convolutional Neural Networks to be established its working is shown in following steps.

CONV 1: Number image of size 32 x 32 x 3, is convolved with 6 filters of size 5 of stride 1 a giving the new image size of 28 x 28 x 6.

MAX 1: We perform max pooling with filter size 2 and stride 2. To get image 14 x 14 x 6

CONV 2: Number image of size 14 x 14 x 6, is convolved with 16 filters of size 5 of stride 1 a giving the new image size of 10 x 10 x 16.

MAX 2: We perform max pooling with filter size 2 and stride 2. To get image 5 x 5 x 16

FC1: All the pixel values of 5 x 5 x 16 i.e. 400 are put into a fully connected layer.

FC2: The previous 400 neuron layer is connected to this new neuron layer of 120 neurons

FC3: The previous 120 neuron layer is connected to this 84 neuron layer.

Output Layer: Final Softmax function having 10 outputs to give between, digits 0 to 9.

This Network in total contains total of **62958** parameters.

## Convolution and Gradient Descent.

We have seen that using convolutions we can drastically drop down the parameters amount from Millions to mere hundreds or Thousands. The additional benefits are:

Parameter Sharing: During convolution the ,multiple filters are being used on the image which can give us many parameters which are learnt by one filter which can used by other filters.

Sparsity of Connections: Each layer only depends upon few input paramaters

We then take a training set of ((x(1),y(1)), (x(2),y(2)), ...... , (x(m),y(m))),

And treat it with Convolutional Neural Network and use the Cost function:

J = m_summation_i=1 {L(y_hat(i),y(i))}

And we use this to perform gradient descent and learn the parameters. 




