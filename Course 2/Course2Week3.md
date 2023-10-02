# Week 3 

## Hyperparameter Tuning

The only major hyperparameter tuning we need to perform is of learning rate, mostly all other hyperparameter already have default values which are widely used in almost all cases. 

We should use random checking of hyperparameters rathe rthan sequential grid pattern follow, this should be done because we should take a broad spectrum of values rather than values which show same pattern, and then we perform coarse to fine approach further narrow down the parameters being used. 

## Panda VS Caviar

Panda method, contains that we look at the cost function of only one model, and then change the hyperparameter and then look at the changes that it made to the cost function, if the function gets smaller, faster then it is a correct tuning to make.

Caviar method, says that we use multiple models with different hyperparameter setting and run all of them simulatneously and then looking at the performances of all the Machine Learning Models we can choose our hyperparameters

## Softmax Regression

Till now we've only seen binary classification, now we can classify the given image into multiple categories this is done by, using softmax regression. It gives output of n dimension, where n is the number of classes and we get output as probabilities of the objects being present or not separately, we then select the image which has highest probability and give that as output. 