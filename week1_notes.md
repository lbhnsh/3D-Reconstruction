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
	
	z= w + b
	
## Cost / Loss / Error function
 **For Logistic Regression**

Loss(error) func:	L(y hat, y) = -(ylog (y hat) + (1-y)log (1 -y hat) )
Cost func:	J(w, b)= Summation from 1 to m { 1/m L(y hat (i), y(i) ) } 

Loss applies to single example, cost for overall
