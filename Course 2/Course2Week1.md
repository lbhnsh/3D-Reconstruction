# Week 1

## Training, Dev and Test sets

Training set are the examples which will be used to train the machine learning model. It is the set which has the largest percentage of data as it gives more features to get extracted and learned by the model 

Development set are the set of example which are used in order to change the algorithm used or just experiment with the model, in terms of layers or activation functions or hyperparameters to get the most optimum model. This is why it is called development set as it develops the model after its training to make it perfect.

Test set, it is the set which as the name suggest is used to test the model, and its accuracy. All the examples are completely new for the model, so we can really get the perfect idea of the accuracy and quality of model.

## Bias and Variance

During training the model can form a learning, which has very less learning to it. This is called the model having **high bias**. 

On the other end, the learing can be extreme on the training set which gives makes it hard for model to idealise to new examples, which gives us poor results. This is called the model having **high variance**

A model to be optimal should have the perfect balance of both bias and variance. 

If the error of training set is high and error of development set is high as well then the model has high bias.

But if the error of training set is very low and error of development set is high then the model has overfitted to the training set which means the model has high variance.


## Basics for modifying Machine Learning Model

If the model has high bias, which we can tell by looking at the error, we have two options either give more time to train for more learning or to increase the number of layers in the network which again increases its learning. 

If the model has high variance, we can give the model more data to learn from which can help idealise due to the new examples introduced. Or we can introduce Regularization which helps weight get sparse and reduce the impact of weights which reduces the overfitting. 


## Regularization for Logistic regression

We know function of cost for logistic regression is 

J(w,b) = (-1/m)*m_summation_(i=1){y\*log(y_hat)+(1-y)\*log(1-y_hat)} + (lambda/2m)\*||w||

We add certain terms to this cost function. There are 2 types of regularization which we can perform,
L1 regularization which uses 

||w||1 = n_x summation j=1(w(j)) 

and L2 regularization which uses

||w||2 = n_x summation j=1(w(j)^2)

**For a Deep Neural Network**

We have the cost function defined as,

J(w[1],b[1],w[2],b[2],....,w[L],b[L]) = (-1/m)*m_summation_(i=1){y\*log(y_hat)+(1-y)\*log(1-y_hat)} + (lambda/2m)\* L summation l=1(||w||^2)

Her we use the forbenius form of regularization ||w[l]|| defined as the summation of all the terms present and its summation.

Now the term which is being added to cost is (lambda/2m)\*(||w||^2), its derivative is (lambda/m)\*(||w||)

So, during backpropogation we can see,

dW[l] = (terms from backprop) + (lambda/m)\*(w[l])

So, W[l] = W[l] - alpha\*(dW[l])

w[l] = w[l] - alpha\*((terms from backprop) + (lambda/m)\*(w[l]))

w[l] = (1- (lambda/m)\*(w[l])) - alpha\*(terms from backprop)


We can see that there is an additional reduction in the order of (1 - (lambda/m)). This is called Weight Decay.

Now this is the reason why regularization works. 

Regularization additionally decays the w[l] which makes parts of the w[l] approximately zero, making the w[l] sparse. Which basically acts as a layer has reduced amount of neurons in it, which simpifies the model, reducing its overfitting.

## Dropout Regularization 

Till now we saw Forbenius form regularization. Now there is another type of regularization which is called dropout regularization. 

This is basically giving a probability to the neurons to a particular layer if the will stay for that iteration or will they be shut down. For example, if the keep_prob is 0.8 and 10 neurons are present in a layer then only 8 neurons will be selected and 2 neurons will be rejected for that iteration. 

The intuition for why dropout regularization works is that by reducing the number of neurons in the layers making it a simpler network.

And overfitting can happen if the neurons are giving more preference to a particular input feature, but dropout gives a random chance for that particular neuron to be present in one iteration which reduces the impact of that input feature on the neuron. Creating evenly spread a little sparse parameter for that layer.

We also need to divide the parameter matrix by keep_prob i.e. a[l]/keep_prob. This is done because we dont want to reduce the whole value, just simplify and modify the impact of parameters. So to compensate for reduction in value due to removal of certain nodes, we divide it by keep_prob.

## Data Augmentation and Early Stopping

Data Augmentation is used when we need to increase the number of training examples by the training examples that we have. We basically create more data from the existing training examples.

For Example if we take a ca picture as an example we can create many more examples by it. We can zoom the picture or zoom it tilting is leftside a bit and rightside a bit, which creates multiple examples from one exmaple. This will not add that much to the learning as a new fresh example can, but it is still a good option to use in cases of scarcity of data.

Early Stopping is the process of stopping the learning iterations early at the minima of cost for development set. 

There are cases, where the cost function will be falling monotonously with training set but suddenly at a particular point, the cost function starts increasing for dev set and keeps normally decreasing for training set. So by stopping at early iteration we get lowst cost function for dev set


## Normalizing Inputs

It is a good practice to normalize the given data before processing it. Normalizaing means setting the example sso that their average is close to zero and their variance is evenly spread. 

We can normalize average by,

n = (1/m)* {m summation i=1 (X(i))}

and X = X - n 

to Normalize variance,

sigma^2 = (1/m)* {m summation i=1 (X(i)^2)}

X = X/sigma^2

This is a good practice as the parameters used usually can have hugely different ranges which can give cost function really elongated and weird shapes giving the model tough time to navigate till the minima. Normalizing makes cost function and evenly spreads the cost function which gives model more ease to find minima.


## Vanishing and Exploding Gradient 

If we look at the model then we can see that the final cost is exponentially dependant on the number of layers present in a particular network. We can then see that they are dependant on each as if the weights are muliplying if we consider the most basic linear function taking place in all layers and no biases being added. 

Because of this even if we consider the weights to be equal and they are just above 1 then it will grow exponentially as the layer keeps on adding which results in an exploded gradient which is huge. Or else if the values are a bit less than 1 the gradient reduces to a great degree creating a vanishing gradient. Which hinders the learning of parameters. '

## Weight initialisation

Till now we've seen that the weights have been initialised randomly but it is better to have some essence of the layer for which it is being created, this can be in the form of multiplying a factor of np.sqrt(1 / n[l-1]). 

## One Vector Checking

We basically combine all the weights and biases into a single array **theta** which concatenates all the weights and biases respectively. After this we perform backward propogation using our model and we get **d theta** of entire model having gradients of weights and biases in order.

Now we can add a very small number let it be epsilon = 10^-7 to **theta** and then subtract the same epsilon from **theta**. Then we can subtract these two arrays and divide it by 2\*epsilon this gives us the **theta_approx**, which has been calculated through trained parameters. 

Techincally the closer **theta_approx** and **theta**, better it is. 

We can check it by the formula 

Check = (||**theta_approx** - **theta**||)/(||**theta_approx**|| + ||**theta**||)

If the value of this Check is around epsilon i.e. around 10^-7 then there is mostly no isse but if there is a big number of Check than epsilon then there is an issue in backpropogation of something 