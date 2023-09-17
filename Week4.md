# Week 4

## Deep Neural Network

We have seen shallow Neural network which have few nodes and one or two layers but for more complex machine learning tasks we normally require a deep neural network which consists of many different layers which have differnt number of nodes having different activation functions, each chosen to make our machine learning model the best for that particular problem. 


The initial input parameter layer is not counted as a layer, and the final layer is not counted as a hidden layer. The hidden layer is the part where the mst learning happens and the final layer is the output layer, which will have sigmoid function or a softmax function.


We know that we denote a parameter of a particular layer with superscript [l] where l is the index of layer in the deep L network. Also we denote input features as a[0], and matrices of other layers as a[1],a[2] .... a[L].

## Why Deep Neural Networks

Deep Neural networks make learning much more effective and intensive. For most of the cases more the number of layers there are more the information can be extracted from the given input feature which is used for more learning. 


For example is we consider a photo to be learned then the inital layers can learn only about the edges with their orientations. They can detect the horizontal edged or vertical edges in the given photo. Then as we go further, we can combine our learning to get more meaningful learning like eyes, ears, mouth or chin etc. And as we go further we can extract more information giving us a complete face. 

## Important blocks for Forward and Backward Propogation

The steps that take place for a particular layer l in forward propogation is 

z[l] = w[l].a[l-1] + b[l]

a[l] = g[l]\(z[l])

So for the forward propogation that particular layer will take input as a[l-1] and will give output of a[l].
We will store the values of w[l] and b[l] in cache, for use in backward propogation

For backward propogation we take input as da[l], and values of w[l] and b[l] from cache. and gives output of 
da[l-1], dw[l] and db[l]. 

Then the value of weights in each layer gets updated as w[l] = w[l] - alpha\*dw[l] and b[l] = b[l]-alpha\*db[l]

## The important steps that occur in Backward Propogation

Given input is da[l],

Then we calculate dz[l] by dz[l] = da[l] * g[l]'(z[l])

And, da[l-1] = w[l].T * dz[l]

dw[l] = dz[l] * da[l-1]

db[l] = dz[l]

Parameters in the neural networks are weights and biases associated with each layer. Hyperparameters are values which manipulate the learning or are used for optimizing things. 