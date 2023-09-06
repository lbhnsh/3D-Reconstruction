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
	
	
<import numpy as np
u=np.exp(v) #each element will be e^v
u=np.log(v)
u=np.abs(v)>

## Vectorizing Logistic Regression
	<X=(nx,m)
	Z=np.dot(w.T,X) +b
	A=[ a1 a2 .... am ] = sigmoid(Z) > 
b is a real number but as it is added with matrix it converts itself into a 1d matrix [b b b... ] which is called **broadcasting**
