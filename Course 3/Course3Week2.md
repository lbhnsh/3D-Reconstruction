# Week 2

## Error Analysis

There can be error due to misclassification in labels. If there is 5 incorrectly labelled in example of 100 then, that contributes 0.5% of the total error. If there is 10% training error we know that 0.5% of it is incorrectly labelled. But there can be other factors which are degrading the error even more. So it wouldn't be a logical decision to change the mislabelled errors. But if thereare 50 mislabeled images in 100 then 5% of the 10% is due to mislabbeled images which can be a perfectly reasonable choice. 

If the percentage contribution of the mislabeled data in the error is big then, we can correct the mislabelled images in the dev as well as test sets.

## Build Your Machine Model Fast and Iterate then

It is an absolute time-saver. We can build the basic model first and then calculate the error from all the different sources. This gives us a huge insight into what is causing error, be it high bias or high variance. And what different methods we need to create to tackle this, be it creating bigger model, traning for longer.

Or introducing Regularization or increase dev set size. Or the error on itself is satisfactory but needs some refining then it can look at what training example it is performing poorly, be it for blurry images and then fix it. 

## Uneven Distributions

Uneven distribution tends to be a simple but rather effective contributor to error causing. This happens because the trained images are from different distribution for e.g. high quality images from internet and the test sets or the objective where it will be actually used is i.e. comparatively low quality mobile photos. This will result in huge errors being produced. 

We can divide the training Example into 4 parts. Training set, Training-Dev set, Dev set, Test set. The training-dev set will be a small subset of the Training set and thats why it will have the same distribution as traning set. Due to this we can get error between different sets, and get more insight. 

If the human error is very less and training error is high then it has avoidable bias. There is very less gap between training error and training-dev error, then it has high variance. If the error between training-dev error and dev error is reasonable then there is data mismatch which means there is different variations for both training and dev sets. 

If we need to address the issue of Data Mismatch, then we need to calculate manual error analysis of all aspects. or else we can just make the training data more similar by mixing the distribution.

## Artificial Data Synthesis

We can synthesize additional training examples. We can take example of speech recognition. So if we used artificial voice then we can layer of noise over it, and make a new example which can create a perfect real life exmaple artificially. But we need to be careful. As if we use the same noise over all the eaxmples then it looks at a very small subset of noise and has trained from it as well, which creates a restricted learning, so we need to do only if needed and that too in a certain range. 

## Transfer Learning

Transfer learning is when we use the already existing learnt parameters of classifier A and put them into classifier B if they have nearly the same objective. The low level features of Classifier A can be helpful for Classifier B and we have trained A with a lot of data.

## Multitask learning

It is the process of making a single model for all the classifications needed. For example rather than creating different classifiers for Stop sign, Pedestrian, Car, Bike separately we can create a single Classifier for all these classifications with a softmax function. 

This is used first to save computational cost and memory, and secondly this wrks because a lot of low level feature layers can be used interchangeably for everything.

End to End deep learning is the process when rather than splitting up the given task into different classifiers sequentially we can directly give an input and train it to give complete output. 

For e.g. for Human detection in Offices using camera, it is mainly divided into 2 steps. First Object Localization takes place and After that another classifier tells if it is an employee or not. Instead of these 2 classifiers we can use one single classifier which does all of these things and outputs the result. Use of end-tp-end classifiers are rarely used in computer vision problems.

These are mainly used in areas where the amount traning data is huge. We need to understnad the complexity of the problem and the hand-skill tuning required for it, and considering this we need to look if we have enough data so it can learn all of these things itself.


