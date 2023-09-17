# Week 3 

## Neural Networks 
Last week we saw the use of only one layer but most of the neural networks which give us helpful results are present in multiple layers. 

If we consider 2 layers then we denote first layer by adding superscript of [1] and second layer with superscript [2].

The input parameters X are given to the network then the steps which will occur are: 

Step 1: Z[1] = W[1].X + b[1]

Step 2: A[1] = sigmoid(Z[1])

Step 3: Z[2] = W[2].A[1] + b[2]

Step 4: A[2] = sigmoid(Z[2])

Step 5: L(A[2],Y) = -(Y\*log(A[2]) + (1-Y)\*log(1-A[2]))

## Neural Network Representation

By convention we dont consider the input layer in the counting of layers for the network. 

If we take a network of only two layers. Then let the 1st layer have 4 nodes and the input features be 3, So each node will have weights associated to then in the shape (3,1). So there are 4 weight column matrices of size (3,1) in layer 1. We convert this to only one weight matrix of shape (4,3). As transpose of the matrix created by stacking weights of individual nodes in columns.

This is then stored in matrix W[1]. The matrix b[1] will have a bias value and will be of the shape (4,1) i.e. the number of nodes in the selected layer.

## Vectorizing across multiple examples

We use multiple examples in order to train our neural network unlike us considering only one example with 3 input features above. If we consider there are n_x input features and if we let number of nodes in a layer 'l' be n[l] then for first layer, the input matrix will "X" will have dimensions (n_x,m). The first layer which will have input X, will have weight matrix (n[1],n_x) which when multiplied with X and added with b[1] gives us matrix Z[1] having dimensions (n[1],m) having linear activation values for all the nodes stacked in columns for different training examples.

Then when we perform a certain function on it, be it sigmoid or tanh or reLU we get A[1], activation values for all the nodes in that layers stacked in columns for different examples. 

## Different types of Activation functions and their importance

There are mainly 3 activation functions that we talk about those are 

**i. Sigmoid**
Sigmoid function is defined as f(z) = 1/(1+e^(-z)). This will give us a graph which will have asymptotes on values y=0 and y=1 and intersects y axis at y = 0.5. This function flattens extremely when the value of z is very large or very small. This is mostly used when we need to perform binary classification and is used in the final layer to give us a probability. If the probability is above 0.5 then the output will be of label 1 and if not then of label 0.

If f(x) is sigmoid function then derivative of the function i.e. f'(x) = f(x)*(1-f(x))
**ii. Tanh**
Tanh function is defined as f(z) = (e^(z) - e^(-z))/(e^(z) + e^(-z)). The graph has asymptotes on y = -1 and y =1 and intesects y axis at y =0. Tanh function is preferred over sigmoid as activation function in layers between the deep L network. This is because the values that we will get will have an average value around 0 unlike sigmoid which will have the average value 0, this gives tanh a plus point in computation and optimization. 

If f(x) is tanh function then derivative of the function i.e. f'(x) = 1-(f(x))^2
**iii. ReLU**
Relu is simple function which is defined as f(x) = max{0,x} . We face a common problem in both tanh and sigmoid that the values get stagnated and are difficult to change when the input value x is either very large or very small. This is removed by reLu function as it does not have any asymptotes but has a linear relation. ReLU function is mostly used as activation function to train neural layers in neural networks. 

If f(x) is ReLU function then, f'(x) for x>0 is 1 and for x<0 is 0.

We sometimes use Leaky ReLU function which is defined as f(x) = max{0.01x,x}

Activation functions are the functions which perform a different function and give neurons for a medium to learn upon. If we only use linear function and perform that on number of deep L layers, then we can just reduce all the layers to one layer and it will create no difference.

## Gradients in a shallow neural network

If we consider a two layer network having first layer of linear activation and second layer of linear sigmoid activation. Them the gradients will be,
dZ[2] = A[2] - Y
dW[2] = (1/m)\*dZ[2]\*A[1].T
db[2] = (1/m)*sum(dZ[2])
dZ[1] = dW[2].T\*dZ[2]\*g[1]'(z[1])
dW[1] = (1/m)\*dZ[1]\*X.T
db[1] = (1/m)\*sum(dZ[1])

## Random Weight initialization

We consider random weight initialization when initializing values of weights intially. We keep the values of the initialized weights small so we multiply them by 0.01. Also if the weights are intialized to 0s then no learning will happen, and nothing will change. But we can initialise bias at zero as it is getting added to give an offset value rather than getting multiplied to x or input features for that particular layer.



