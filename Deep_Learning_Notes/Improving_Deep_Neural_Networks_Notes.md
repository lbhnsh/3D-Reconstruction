**Applied ML is a highly iterative process**
#WEEK 1:
##Train/ Dev/ Test sets
Parameters to consider while developing a NN:
1. no. of layers
2. no. of hidden units
3. learning rates
4. activation functions

Standard distribution for moderate data size:
60 : train
20: dev
20: test

For Bigger data set say more than a million:
even 1 % to dev and 1% test can be enough
i.e more the number of data size the % of train can be higher

MAKE SURE *dev* and *test* come from the same distribution

Human error in classification considered to be 0%
This is 
Optimal (Bayes) error approx 0% error


IF HIGH BIAS (training set performance) -> Make Bigger network, train larger, NN architecture search
IF NO but HIGH VARIANCE (dev set performance)1->  more data required, regularization, NN architecture search

## Regularization:

When Training size can't be increased we apply Regulartization

#### Consider Logistic Regression

Cost(w,b)=(1/m)*np.sum(loss(yhat,y)) + Regularization term


###### L2 Regularization: 

Regularization term= (lambd/2m)* ||w||^2 ...[subscript 2]
L2 Regularization : ||w||^2 = np.sum(wj)^2 = np.dot(w.T,w)

###### L1 Regularization: 

Regularization term= (lambd/2m)* np.sum(|wj|) ...[subscript 1]
L1 Regularization : ||w||


There is also a Regularization term for bias part, but it is not much effective.



	||.||^2    ....[subscript 2]
**This is called the Frobenius Norm**


lambda is a hyperparameter
lambda is a default function so use the word lambd

#### Regularization During Back Propagation is called WEIGHT DECAY

Weight decay term is the middle part of the equation

W= (1- (learning_rate*lambd)/m)*W - learning_rate*dW






##### How does Regularization prevent Overfitting?
As weights are brought to near zero, the overfitting (high variance) part becomes a case of High bias as Weights are approximated to zero the model acts similar to a logistic regression model






### Dropout Regularization
some part of the neurons are assigned zero value and rest are assigned 1

1. d3=np.random.rand(a3.shape[0],a3.shape[1]< keep_prob
keep_prob generally between 0.5 and 1.0

what keep_prob does is randomly it initializes say 0.8 part of a layer to 1 and 0.2 part to 0 

2. a3=np.multiply(a3,d3)

3. a3/= keep_prob

so the values regulate the reduction of neurons


### Other Regularization Methods:


### Setting Up Optimization problem
#### Normalizing Inputs:
1.Subtract out or zero out the mean
	x=x-mu

2.Normalize the variance
	x/= sigma

If inputs are unnormalized cost function will be very elongated
but if they are normalized it will look more symmetric

#### Vanishing/ Exploding Gradients

For deep neural networks the activation values can go much higher so we set the weights to less than 1 so that increasing their power wont increase them


#### Weight Initialization for deep networks

Larger the num of layers smaller the W i

W^[l]=np.random.rand(shape)*np.sqrt(2/n^[l-1])
Var(w:) =2/n

Other variants:
FOR tanh:

sqrt(1/n^[l-1])               <-- Xavier initialization


#### Numerical Approximation of Gradients

Checking your derivative computation

g(theta)=(f(theta+epsilon)-f(theta-epsilon))/2(epsilon)

#### Gradient Checking (J=cost)

Take all parameters W[l],b[l] and reshape into a big vector Theta
Take all parameters dW[l],db[l] and reshape into a big vector dTheta

Grad check 

for each i:
	dTheta(approx)[i]= (J(Theta1+Theta2+.....+ Theta i + Epsilon,...) - J(Theta1+Theta2+.....+ Theta i - Epsilon,..))/(2*Epsilon)

			approx= dTheta[i]= delJ/delTheta i

Check should be around 10^-7---> ratio of eucliadian distance


#### Gradient checking implementation notes
1.Don't use in training onlt to debug
2.If algo fails grad check, look at W,b etc. to try to debug
3.Remember regularization
4.Doesn't work with dropout


---------------------------------------------------------------------------

#WEEK 2:
##OPTIMIZATION ALGORITHMS:
To speed up 

### Mini-Batch Gradient Descent
 Vectorization allows to efficiently compute on m examples
Split training set into mini-batches

X^{t} for t th mini-batch 
dimensions: (nx,1000)
Y^{t} for t th mini-batch 
dimensions: (1,1000)

1000 is an arbitrary number, can be anything

**L has been used previously to indicate the number of the layer in the network, in this particular example was also used to indicate the number of samples in the mini-batch.**

for t=1 to 5000
	Forward prop on X{t}
		Z=WX{t}+b
		A=g(Z)....(apply vectoization for forward prop)
	Compute cost J
	Backprop to compute gradients wrt J using X{t},Y{t}
	W[l]=W[l]-learning_rate*dW[l]
	b[l]=b[l]-learning_rate*db[l]

### Understanding Mini-Batch Gradient Descent

* Cost maynot decrease in every batch
**Mini batch is a hyperparameter**

Choosing mini batch size:
   Stochastic descent size: each batch has size 1
   Batch descent size: only one batch of size m

If small training set use Batch gradient descent
Typical mini-batch size: 64,128,256,512..


###Exponentially weighted Averages
v(t)=Beta*v(t-1) + (1-Beta)*Theta(t)

v(100)=0.1*Theta(100) +0.9*0.1*Theta(99) +0.9*0.9*0.1*Theta(98)+....
 
#### Bias Correction in Exponentially Weighted Averages
V(0)=0 
to get a better curve


### Gradient Descent with Momentum:
MOMENTUM:
On iteration t:
	Computee dW,db on current mini-batch
	V(dW)=Beta*V(dW) + (1-Beta)*dW
	V(db)=Beta*V(db) + (1-Beta)*db
	W=W-learning_rate*V(dW)
	b=b-learning_rate*V(db)
Hyperparameters: learning_rate, Beta ..... Beta=0.9 works well in practice

V(dW), V(db) -> 0

(1-Beta)*dW is ignored
(1-Beta)*db is ignored


### RMSprop (ROOT MEAN SQUARE):

On iteration t:
	Computee dW,db on current mini-batch
Small no:	S(dW)=Beta*S(dW) + (1-Beta)*(dW^2)...........squaring is element wise
Large no:	S(db)=Beta*V(db) + (1-Beta)*(db^2)
		W=W-learning_rate*(dW/sqrt(S(dW))
		b=b-learning_rate*V(db/sqrt(S(db))


*we can add epsilon to the RMS part*


### Adam Optimization Algorithm:

V(dW),S(dW),V(db),S(db)=0
On iteration t:
	Computee dW,db using current mini-batch
	V(dW)=Beta*V(dW) + (1-Beta)*dW
	V(db)=Beta*V(db) + (1-Beta)*db
	S(dW)=Beta*S(dW) + (1-Beta)*(dW^2)
	S(db)=Beta*V(db) + (1-Beta)*(db^2)
	V(dW)corrected=V(dW)/(1-Beta(1){t})
	V(db)corrected=V(db)/(1-Beta(1){t})
	S(dW)corrected=S(dW)/(1-Beta(2){t})
	S(db)corrected=S(db)/(1-Beta(2){t})
	W=W-learning_rate*V(dW)corrected/sqrt(S(dW)corrected+epsilon)
	b=b-learning_rate*V(db)corrected/sqrt(S(db)corrected+epsilon)

Hyperparameters:
learning_rate: needs tuning
Beta1 = 0.9
Beta2= 0.999
epsilon = 10^-8

**Adam: Adaptive Moment Estimation**



### Learning Rate Decay

Slowly reducing learning_rate helps in oscillating in the minima part, if its not done the machine keeps learning (oscillating) and may go away from the minima

**learning_rate = learning_rate0 / (1+ decayRate*epochNumber)**

1 epoch = 1 pass through data


|Epoch|learning_rate|
|-----|-------------|
|    1|		 0.1|
|    2|	       0.067|
|    3| 	0.05|
|    4|         0.04| 

Other Learning_rate decay methods:

learning_rate = (0.95^epochNumber)*learning_rate0 	<---Exponential decay
learning_rate = (k/sqrt(epochNumber))*learning_rate0	<---Discrete Staircase
--k is some constant

OR Manually decay it


###The Problem of Local Optima

Problem of Plateaus: Gradient is around 0 for a long period
Which is unlikely 
But Plateaus can make learning slow


-------------------------------------------------------------------------------------------

# WEEK 3:
## Hyperparameter Tuning

### Tuning Process:

Hyperparameters:
There are a lot of hyperparameters that require tuning 
 1. Try Random values: Don't use a grid
 2. Coarse to fine: Take a smaller set and try in that smaller grid to find random values


### Using an Appropriate Scale to pick Hyperparameters

~Picking hyperparameters at random~

Appropriate scale for hyperparameters

r=-4*np.random.ran()  <-- r : [-4,0]
learning_rate=10^r

Hyperparameters for Exponentially Weighted Averages:
say beta
r:[-3,-1]
beta=1-10^r

### Hyperparameters Tuning in Practice: Pandas vs Caviar
Re-test hyperparameters occasionally
1. Pandas approach: Babysit one model
2. Caviar approach: Train many models in parallel

**REFERENCE TO WEEK 1: Normalization should be: X= X/sigma**
**Please notice that inputs should be normalized dividing by sigma and not sigma squared.**

Normalize Z[2] to fasten computation of W[3] 
We only normalized the input layer in WEEK1's part

BATCH NORMALIZATION is implementing normalization to Hidden Layers as well

Given some intermediate values in NN 
	mu=(1/m)* np.sum(Z(i))
	sigma=(1/m)*np.sum(Z(i)-mu)
	Z(i)norm= (Z^(i)- mu)/(sqrt(sigma+epsilon))
	Z~(i)norm= gamma*Z(i)norm + beta

	if 
	gamma=sqrt( sigma+ epsilon )
	beta=mu
	then	Z~(i)=Z(i)

	(gamma and beta are learnable parameters of model)

### Fitting Batch Norm to a Network
Parameters: W[l],b[l], gamma , beta 
note: this beta is not of the hyperparameter

We work on mini-batches
X{1} -> Z[1] -> Z~[1]-> a[1]-> Z[2]....

**When applying Batch Norm b[l] can be excluded (it has no point)**

#### Implementing Gradient Descent

```
for t=1 to num_minibatches
	Compute forward prop on X{t}
		In each hidden layer, use Batch Norm to replace Z[l] with Z~[l]
	Use Backprop to compute dW[l], dbeta[l], dgamma[l]
	Update params (subtraction of learning_rate*dParameter)
	# Works with RSM, Adam, Momentum
```

### Why does Batch Norm work?

* Learning on Shifting Input distribution
**"Covariate Shift"**


#### Batch Norm as Regularization
1.Each mini-batch is scaled by the mean/variance computed on just that mini-batch
2.This adds some noise to the values z[l] within that minibatch. So, similar to dropout, it adds some noise to each hidden layer's activations.
3.This has a slight regularization effect.


### Batch Norm at Test time

	mu=(1/m)* np.sum(Z(i))
	sigma=(1/m)*np.sum(Z(i)-mu)^2
	Z(i)norm= (Z^(i)- mu)/(sqrt(sigma^2+epsilon))
	Z~(i)norm= gamma*Z(i)norm + beta

Estimate mu,sigma using exponentially weighted average(across minibatches)


## Multi-Class Classification

### Softmax Regression
Y layer is (n[L],1)


Softmax Activation Function:
t=e^z^[l]
a[l]= e^z^[l] / sum of t


### Training Softmax Classifier:

1.Loss(yhat,y)= -np.sum(y*logyhat)

2.Cost=1/m np.sum(L(yhat,y))

3.Gradient Descent:
Backprop: dz[l]=yhat-y


## Introduction to Programming Frameworks
### Deep Learning Frameworks:
Keras,Torch,TensorFlow

### TensorFlow 
