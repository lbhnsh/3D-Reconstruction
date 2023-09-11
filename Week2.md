# Week 2
## Binary Classification
Binary Classification is the terminology used when the neural network has to decide between 2 choices. For example it can be a cat or NOT a cat image.

It depends on the input data we give to the neural networks, Now the size of the input image in array is going to be (n_x,m), where n_x is the number of inputs and m is the number of traning examples present in the given training set.

Similarly we have an output set which will have the dimension of (1,m), where array of dimension 1 is because it will contain one output which is 0 or 1 i.e. true or false for the given input and m is the number of training examples.

## Logistic Regression
Logistic regression is the method which is used to give a input and we will get the answer in terms of probability. If the probability given i.e. y_hat greater than 0.5 then we will say it is 1 i.e. the output will be true or else false.

In symbolic notations is given by 
 𝑥 , 𝑦̂ = 𝑃(𝑦 = 1|𝑥), where 0 ≤ 𝑦̂ ≤ 1
z = wT . x + b
We use sigmoid function which is given by 
y_hat = sigmoid(z) = 1/(1+e^-z)
to calculate the probability.

Each neuron is assigned which parameters namely weight and bias. Weight is the size of the input and it contains values which can correspond to each input parameter and a bias to be added in the to the matrix multiplication of weight transpose(Wt) and input x, which will be called Z i.e. output of linear operation

And then we will use our activation function, in this case sigmoid on Z to get the final output.

## Loss function and Cost Function
**Loss function** is extremely important in order to calculate the accuracy of our model, the loss function gives us the comparision between the calculated probability of the neural network and the actual output of the training example. Less the value of loss function is greater the probability will be to guess the correct output.

Loss function is defined as 
L(y_hat,y) = -(y\*log(y_hat)+(1-y)\*log(1-y))

Loss function is used to calculate the loss of just one training example. If we need to calculate the accuracy for the entire training set consisting of m training examples, we will require a **cost function**

Cost i.e. J(w,b) = (-1/m)*m_summation_(i=1){y\*log(y_hat)+(1-y)\*log(1-y_hat)}

## Gradient descent
The cost function will will be plotted for w and b will have a convex base. The original w and b defined by us will be arbitrary, now with the help of gradient descent we need to change the values of w and b to a point where it will be close to the global minimum of the graph. 

We do this with the help of derivative of J(w,b) w.r.t to w and b which is called dw and db on every step of the gradient descent.
And we update the values of w and b according to the following

w := w-alpha*(dw)
b := b-alpha*(db)

where alpha is the learning rate and decides the speed at which the updation of values of w and b is going to take place. 

## Logistic Regression Gradient descent

There are going to be 4 major steps for logistic regression. 
The input parameters present with us are x1,x2,w1,w2,b.

Step 1: Input paramters declaration of x1,x2,w1,w2,b.
Step 2: z = w1.x1 + w2.x2 + b
Step 3: y_hat = a = sigmoid(z)
Step 4: L(a,y) = -(y\*log(a) + (1-y)\*log(1-a))

Now if we backpropogate then for step 2 we need to calculate derivative of L(a,y) in step 3 w.r.t. a in step 2.
Which will be dJ(w,b)/da = -(y/a) + (1-y)/(1-a)

For backpropogating to Step 1, we will calculate derivative of L(a,y) w.r.t. z.
which will be dJ(w,b)/dz = (-(y/a) + (1-y)/(1-a))*(a\*(1-a)) = a-y

And finally for upgrading the original parameters we will now derivate them w.r.t. cost function.
dw1 = dJ(w,b)/dw1 = x1*dz
dw2 = dJ(w,b)/dw2 = x2*dz
db = dJ(w,b)/db = dz

And now when we have the derivatives with us we can update our weights so that we can get close to the global minimum of the cost function with operations

w1 := w1-alpha*(dw1)
w2 := w2-alpha*(dw2)
b := b-alpha*(db)

We looked at this example of only one training example and in reality we use m number of training examples, and to apply gradient descent to all these need to use a for loop which will iterate through all the examples one by one. And in practice we use a whole lot of weights rather than 2. Therefore we need to put an additional for-loop to update n number of weights for m number of training examples.

The downside is that we should not use for-loop for training neural networks as it takes a huge amount of time to process and requires high computation, which becomes much more apparent when we use huge datasets. 

## Vectorization

We use vectorization in order to bypass the obstruction of using for-loop for our training
For vectorization we store the weights and biases in arrays and we use library numpy in order to operate matrix multiplication on those matrix.

When using time class, we can easily look that the time required for performing same task through for-loop can take upto 30 times the time required to perform the same task by vectorization i.e. numpy functions.

There are many functions that numpy library carries. If we take two matrices a and b then,
np.dot(a,b) does matrix multiplication of a and b
np.subtract(a-b) performs subtraction
np.exp(a) gives matrix which contains e to the power of the value stored in matrix
np.log(a) gives matrix which has log of all the values in that matrix.

## Vectorization for logistic regression 

Now if we consider input parameters for a single training example then the size of the matrix x will be (n_x,1) where n_x is the number of input parameters and 1 because of 1 training example. 

We create a matrix 'X' which has all the training examples stacked in columns so its size will be (n_x,m)
where n_x is the number of input parameters and m is the number of training examples. 

Now as weights corresponds to all the input parameters in each column, so it will be of size (n_x,1) which will be called 'W'. One bias will correspond to each value after matrix multiplication of W transpose and X, so it will have size (1,m). Y is matrix which has actual output to all the training examples, it will have size (1,m)

Now to calculate the products using vectorization we use 
Z = np.dot(W.T,X) + b
A = sigmoid(Z)

Here Z is of dimensions (1,m) and A is of dimensions (1,m)

__**Broadcasting**__
Now we couls've used only single value of b rather than a matrix, python itself would've created its array to the size of the matrix to which it will be added. 
This particular feature is called Broadcasting

## Vectorization for Gradient Descent

We've calculated A in the previous topic using forward propogation and from the topic of gradient descent 
we know dz = a - y.

So now using it on matrices of m examples we can say,
dZ = A - Y 
dZ will have same dimensions as A and Y i.e. (1,m)

Now we can calculate X.dZ transpose for all examples using for loop and keep on updating dW by adding all the values to it and then dividing it by m at the end, but we will be using vectorization. 

So we can easily write, 
dw = (1/m)*np.sum(np.dot(X,dZ.T))
db = (1/m)*np.sum(dZ)

