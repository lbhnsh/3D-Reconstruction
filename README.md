# 3D-Reconstruction
3D Reconstruction from Single RGB Image
# Week 1

__Neural network__ is an interconnected web of a number of functions, where output from one function serves as input for next function, leading to a common output. Neural network can be viewed as a machine that takes data set as input and returns output after processing at multiple levels.
___
1. Structured data
   |Size(sq.ft)|No. of bedrooms|...|Price($)|
   |-----------|---------------|---|----------|
   |2104|3| |400|
   |1600|3| |330|
   |...|...| |...|
   |3000|5| |780|
2. Unstructured data
   ![Example : Image, Audio, Text etc](https://1drv.ms/i/s!Ao0tbw-VcjB7hoQK_RtZjM6BHtUfjw?e=2YAynS)
---
As data is considered fuel of NN, as the amount of data in input dataset increases, performance of NN increases upto some limit(for traditional algorithms). Performance increases more for larger NNs. 

![in2](https://www.researchgate.net/profile/Carlos-Dominguez-Monferrer/publication/365133696/figure/fig1/AS:11431281106808049@1670854952375/Scale-drives-Machine-Learning-and-Deep-Learning-progress-Adapted-from-63.png)

# Week 2
__Binary Classification__
(x,y) where x ∈ R<sup>3</sup> and y ∈ {0,1}
If we have m examples in training set, then
X={(x<sup>(1)</sup>,y<sup>(1)</sup>),(x<sup>(2)</sup>,y<sup>(2)</sup>),.....................(x<sup>(m)</sup>,y<sup>(m)</sup>)}
X is a matrix of n<sub>x</sub> rows and m columns
```
X.shape()  #(nx,m)
```
___

__Logistic Regression__

For X as input we want y^ as output. y^=P(y=1|x)
i.e   0<=y^<=1
Parameters : w ∈ R<sup>n<sub>x</sub></sup>, b∈R
Output: y^ = σ(w<sup>T</sup>x + b)
let   w<sup>T</sup>x + b  =  z
σ(z) is sigmoid function expressed as 1/(1+e<sup>-z</sup>)
![graph](https://www.researchgate.net/profile/Atsushi_Teramoto/publication/336275807/figure/download/fig2/AS:810595121111040@1570272345204/Standard-sigmoid-function.ppm)
---

Loss function

L(y^,y) = 0.5 * (y^-y)<sup>2</sup>
        = -(ylogy^ + (1-y)log(1-y^))

Cost function

J(w,b) = (1/m)*∑<sub>i=1</sub><sup>i=m</sup>L(y^,y)
we want y^ closest to y
---

That means we want w and b that mainimize J(w,b)

__Gradient Descent__

Derivative of J(w,b) w.r.t w or b is called  dw or db respectively. α is constant called learning rate.  On repeated 
w = w - αdw         and         b = b - αdb
we get minima point
---

__Computation Graph__

![example](https://th.bing.com/th/id/R.1f8820dd78836a0fcafcbfc3575be189?rik=q%2b6kX%2bfO5YmWtQ&riu=http%3a%2f%2fmedia5.datahacker.rs%2f2018%2f06%2fword-image-18_new.png&ehk=7%2bXJ13CdqFKr3DKyrxQyG1VS01C1dpP%2fUSgY9Y9iLsI%3d&risl=&pid=ImgRaw&r=0)
dj/du = dj/dv * dv/du
dj/db = dj/du * du/da
dv/db = dv/du * du/db
---
Brief summary
z = σ(w<sup>T</sup>x + b)
y^ = a = σ(z)
L(a,y) = -(ylog(a)+(1-y)log(1-a))
___

__Logistic Regression derivatives__
z = w1x1 + w2x2 + b
a = σ(z)
L(a,y)
|dz = dL(a,y)/dz  |  da = dL(a,y)/da|
|-----------------|-----------------|
|   = dL/da * da/dz|    = -(y/a)+(1-y/1-a)|
|   = a-y          |
---
w1 = w1 - αdw1
w2 = w2 - αdw2
b = b - αdb
---
for m examples,
J(w,b) = (1/m)*∑<sub>i=1</sub><sup>i=m</sup>L(a<sup>(i)</sup>,y<sup>(i)</sup>)
a<sup>(i)</sup> = y^<sup>(i)</sup> = σ(z<sup>(i)</sup>) = σ(w<sup>T</sup>x + b)

d J(w,b) /dw1 = 1/m * ∑<sub>i=1</sub><sup>i=m</sup> L(a<sup>(i)</sup>,y<sup>(i)</sup>)

___
__What is vectorization ?__
          z = w<sup>T</sup>x + b  
Non vectorized      
z=0
 ```
 for i in range(n-x):
     z+=w[i]*x[i]
 z+=b
 ```
Vectorized
```
z=np.dot(w,z)+b
```
(z=w<sup>T</sup>x + b)

Vectorization is implemented with an aim of eliminating explicit for - loops

Example of vectorization
Suppose we have to apply exponential operation on every element of matrix
v=$\begin{bmatrix}v1\\v2\\.\\.\\.\\vn \end{bmatrix}$ we want u = $\begin{bmatrix}e^v1\\e^v2\\.\\.\\.\\e^vn \end{bmatrix}$

Unvectorized method
```
u=np.zeros((n,1))
for i in range(n):
    u[i]=math.exp(v[i])
```
Vectorized method
```
import numpy as np
u = np.exp(v)
```
Similarly,
```
np.log(v)
np.abs(v)
```
___

__Vectorizing Logistic Regression__
z<sup>(1)</sup> = w<sup>T</sup>x<sup>(1)</sup> + b
a<sup>(1)</sup> = σ(z<sup>(1)</sup>)

z<sup>(2)</sup> = w<sup>T</sup>x<sup>(2)</sup> + b
a<sup>(2)</sup> = σ(z<sup>(2)</sup>)

...

z<sup>(m)</sup> = w<sup>T</sup>x<sup>(m)</sup> + b
a<sup>(m)</sup> = σ(z<sup>(m)</sup>)

Z = $\begin{bmatrix}z^1 z^2 ... z^m \end{bmatrix}$
A = $\begin{bmatrix}a^1 a^2 ... a^m \end{bmatrix}$ = σ(Z)

dz<sup>(1)</sup> = a<sup>(1)</sup> - y<sup>(1)</sup>
dz<sup>(2)</sup> = a<sup>(2)</sup> -y<sup>(2)</sup>
dZ = $\begin{bmatrix}dz^1 dz^2 ... dz^m\end{bmatrix}$
dZ = A-Y = $\begin{bmatrix}a^1-y^1\ a^2-y^2\ ... a^m-y^m  \end{bmatrix}$

db = (1/m) * np.sum(dZ)
dw = (1/m) * X*dZ<sup>T</sup>
   = (1/m) * $\begin{bmatrix}x^1 ... x^m\end{bmatrix}$ * $\begin{bmatrix}dz^1\\.\\.\\.\\dz^m\end{bmatrix}$
   = (1/m) * $\begin{bmatrix}x^1dz^1 + ... + x^mdz^m\end{bmatrix}$

dw=0
dw+=x<sup>(1)</sup>dz<sup>(1)</sup>
dw+=x<sup>(2)</sup>dz<sup>(2)</sup>
...
upto m

db=0
db+=dz<sup>(1)</sup>
db+=dz<sup>(2)</sup>
...
upto m

___

__Implementation of Logistic Regression__

J=0, dw1=0, dw2=0, db=0
for i = 1 to m:
    z<sup>(i)</sup> = w<sup>T</sup>x<sup>(i)</sup> + b
    a<sup>(i)</sup> = σ(z<sup>(i)</sup>)
    J += -[y<sup>(i)</sup>loga<sup>(i)</sup> + (1-y<sup>(i)</sup>)log(1-a<sup>(i)</sup>)]
    dz<sup>(i)</sup> = a<sup>(i)</sup> - y<sup>(i)</sup>
    dw<sub>1</sub> += x1<sup>(i)</sup>dz<sup>(i)</sup>
    dw<sub>2</sub> += x2<sup>(i)</sup>dz<sup>(i)</sup>
    db += dz<sup>(i)</sup>
J=J/m
dw<sub>1</sub> = dw<sub>1</sub>/m
dw<sub>2</sub> = dw<sub>2</sub>/m
db = db/m
___

__Broadcasting__
General Principle

(m,n) matrix    op    (1,n) matrix  =  (m,n) matrix
(m,n) matrix    op    (m,1) matrix  =  (m,n) matrix

where op can be +,-,*,/


