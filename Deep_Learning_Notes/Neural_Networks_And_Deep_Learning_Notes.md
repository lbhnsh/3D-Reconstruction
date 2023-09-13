# Introduction to Deep Learning

## Neural Networks

**The remarkable thing about neural networks that have been given enough data about x and y, given enough training examples with x and y, neural networks figure out accurate functions to map x and y**

Standard NNs for simpler things
CNNs for images
RNNs for sequential data
Custom/ Hybrid for complex things

|Structured Data | Unstructred Data|
|----------------|-----------------|
|datasets        |images, words    |

*·m : number of examples in the dataset. x : input label(feature vector). y : output label. nx= dimension of input features x*


forward pass & backpropagation


## Binary Classification

Training set -> (x,y)
x belongs to nx dimensions
and y is 0 or 1
m= no. of training examples i.e {(x^1,y^1),(x^2,y^2),...(x^m,y^m)}

Training set denoted by X
X is a matrix with m columns and nx rows

( **X.shape for finding shape of matrix** 
i.e X.shape=(nx,m) )


Stack(verb) Y in columns
Y=[ y^(1),y^(2),...., y^(m) ]
(one dimensional matrix)

( **Y.shape for finding shape of matrix** 
i.e Y.shape=(1,m) )

## Logistic Regression 
**Logistic Regression is an algorithm for Binary Classification** 

	Let for training example Logisitc Regression have output y hat (y^) which is a probability i.e. range 0 to 1
	If we use Linear Regression the output can be greater than 1 or negative.
	The parameters are w and b
	
We use **Sigmoid** function for Logistic Regression
! [Sigmoid: ] (https://miro.medium.com/v2/resize:fit:640/format:webp/1*aQPjVStYupSRGtIsjq3adA.png)
	
	z= wx + b
	
## Cost / Loss / Error function
 **For Logistic Regression**

Loss(error) func:	L(y hat, y) = -(ylog (y hat) + (1-y)log (1 -y hat) )
Cost func:	J(w, b)= Summation from 1 to m { 1/m L(y hat (i), y(i) ) } 

Loss applies to single example, cost is average of all loss

## Gradient Descent

To find w and b that minimize J(w,b)
J is a convex function

1. Choose a random point of initialization
2. Finds Global optimum 
3. Repeats w:= w - alpha [ dJ/ dw]
   b:= b - alpha [ dJ/ db]  .... alpha= learning rate

### Computation graph:
One step of **backward** propagation on a computation graph yields derivative of final output variable.
### Computing derivatives:
	While coding the computing derivatives:
	dJ/dv --> dv
	dJ/da --> da where v and a are variables of previous step(s)
Computing derivatives is easier RIGHT to LEFT

### Logistic Regression Gradient Descent

z= w1x1+w2x22+b
dz=a-y		a=sigmoid(z)=y hat

dw1=x1.dz 
dw2=x2.dz
->imp lect: (https://www.coursera.org/learn/neural-networks-deep-learning/lecture/udiAq/gradient-descent-on-m-examples)

**Vectorization Techniques Avoid For Loops (it is a faster method)**
(numpy lib)

z=np.dot(wT,x) +b 
	*np.dot(wT,x)=wT x* 
	
```	
import numpy as np
u=np.exp(v) #each element will be e^v
u=np.log(v)

 

u=np.abs(v)>
```
## Vectorizing Logistic Regression
```
	X=(nx,m)
	Z=np.dot(w.T,X) +b
	A=[ a1 a2 .... am ] = sigmoid(Z) 
```
 
b is a real number but as it is added with matrix it converts itself into a 1d matrix [b b b... ] which is called **broadcasting**

```
	dz=a(i).y(i)
	db= 1/m np.sum(dz)
	dw= 1/m X dz(transpose)
```

**Without using any for loops:** Let ' be Transpose
```

Z=w'X+b	=np.dot(w.T,X) + b
A=sigmoid(Z)
dz=A-Y
dw=1/m X dz'
db=1/m np.sum(dz)
w:=w- alpha dw
b:=b- alpha db

```

## Broadcasting
	when python automatically matches one dimension
axis=0 (vertical direction)
axis=1 (horizontal direction)

## Neural Network Representation

|INPUT layer | Hidden Layer | OUTPUT LAYER |
|------------|--------------|--------------|
|a^[0]=X     |a^[1]         |a^[2]         | 

this is a 2 layer neural network


Superscript is for layer number
Subscript for node number in layer


## Why do we need non linear activation functions?


In one node there are two parts (considering logistic regression)
. Left part calculates z=dot(w.T,x)+b and right part calculates sigmoid(z)


|Given input x: | 	       |
|---------------|--------------|
|z^[1]=W^[1]x +b^[1]|a^[1]=sigmoid(z^[1])|
|(4,1)  (4,3)(3,1)  (4,1)|(4,1)  (4,1)| 
|z^[2]=W^[2]x +b^[2]|a^[2]=sigmoid(z^[2])|
|(1,1)  (1,4)(4,1)  (1,1)|(1,1)  (1,1)|

## Formulas for computing derivatives:
### Forward Propagation:
	Z^[1]= W^[1]X + b^[1]
	A^[1]= g^[1] (Z^[1])
	Z^[2]= W^[2]A[1] + b^[2]
	A^[2]= g^[2] (Z^[2])= sigmoid(Z^[2])
	
### Back Propagation:
	dZ^[2]=A^[2]-Y
	dW^[2]=1/m dZ^[2] A^[2].T
	db^[2]=1/m np.sum(dZ^[2],axis=1,keepdims=True)
	dZ^[1]=( W^[2].T dZ^[2] )*( derivative of g^[1] (Z^[1]) )
	dW^[1]=1/m dZ^[1] X.T
	db^[1]=1/m np.sum(dZ^[1],axis=1,keepdims=True)
	
### L Layered Neural Network 
Logistic Regression -> Shallow neural network
Previous -> 2 layer Neural network
More layers-> Dense neural network

L= no. of layers
n^[l] = no. of nodes in layer no. l 
e.g: n^[1] = no. of nodes in first layer
W[l] = weights of Z[l]

### Forward Propagation in deep layer neural network

####### (here l is small L) 
generalized formula:
      **Z^[l] = W^[l]. A^[l-1] + b^[l]**
where A^[0] = X

To iterate Z from l=1 to l=L we need to use FOR loop which is unavoidable


### Dimensions of Matrix 

W[l] :        .. (n^[l], n^[l-1]) 
b[l] :             (n^[l], 1) 
dW[l] and db[l] should have same dim as W and b respectively

Z[l] :: A[l] :     (n^[l], m) 
(when vectorized) 
when vectorized b[l] : (n^[l], m) but as it is originally b[l] :  (n^[l], 1) it gets broadcasted m times

same dimensions for dZ[l] and dA[l]

dA and other derivatives are stored in cache

dZ[L]=A[L]−Y

dW[L]=1mdZ[L]A[L−1]T

db[L]=1mnp.sum(dZ[L],axis=1,keepdims=True)

dZ[L−1]=W[L]TdZ[L]∗g′[L−1](Z[L−1])

Note that * denotes element-wise multiplication)

⋮

dZ[1]=W[2]TdZ[2]∗ g′[1] (Z[1])

dW[1]=1mdZ[1]A[0]T 

Note that A[0]T is another way to denote the input features, which is also written as XT

db[1]=1mnp.sum(dZ[1],axis=1,keepdims=True)


###Parameters:
W[l] b[l]
###Hyperparameters:
iterations, learning rate, etc. that help the parameters

