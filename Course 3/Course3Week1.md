# Week 1

## Orthogonalisation

Orthagonalization is the process of keeping things separated, And work on handling differnet things differently. We can consider it like handling speed and direction with two different things i.e. Accelerator and Steering wheel. We can combine everything into a joystick and move it around at particular angle and particular height to denote differnt acceleration and direction but this just makes thing more difficult and hard to understand.

Similarly we can keep things separately in Machine Learning. We can keep things like, Training an example as one step and Optimizing the training example as another step. Rather than mixing everything together and getting confused as changing what can affect what. 

## Single Number Evaluation Metric

Usually there are multiple parameters be it error calculation in two cases, One the images it read as cats we cats. And second out of how many cat images did it correctly identify as cats. We have both values for 2 classifiers.

Rather than looking at these two values separately and being confused about their priorities we can combine them into one metric called F1 score in this case which will take harmonic mean of the two errors being calculated. This will give us a more direct and easy way of judging the classifiers.

## Satisficing and optimizing Metrics

Most of the time the parameters given about the classifier wont be errors which can be correlated together. Most of the time there are different things related to a classifier such as Running Time or computational space which is present. 

In this case we take the metric with the highest priority, which will be error. Or number of errors associated with one classifier. This will be called Optimizing Metric and we can just put a threshold on other metrics for e.g. if the runnign time is below 25 ms then the classifier is good enough or the memory taken snould be 2 GB then its good enough.

## Train, Test, Dev Sets

We should insure that the data in all the sets being created should originate from the same distribution. Or else there can be cases that we've trained the image on high level images present on the web and the test data are of a bit low quality mobile images then it will perform badly as these mobile images aren't present in the training set.

Size of the sets depend upon the amount of training examples which we have. If the data is relatively less then we can use a 70-30 split. Or if we have huge amount of data say 1M, then we can use 98-1-1 split. The Dev and Test set should be big enough to detec the difference between different models that we're trying out. 

There can be instances when the classifier with smallest error isn't good. If we consider the a ML which gives out CAT images. There is A which gives images with 3% error and B which gives 5% error. But if the classifier A also gives a lot of obsence adult imagery then it is unacceptable according to social norms, thats why we'll prefer B over A.

## Human Level Performance

There is a certain accuracy which Human's can achieve which is called Human-Level performance. There is a accuract which surpasses Human Level Performance, error associated with that accuracy is called Baye's Error and it is the least possible error that can be achieved by the ML model even if it surpasses Human Error.

If the gap between the training error and Human error is small then it is called Avoidable Bias i.e. with more training that error can be surpassed but after a clear value of Baye's error is unknown which slows down the learning process after surpassing human level performance.

We can use Human Error as proxy of Baye's Error as the gap between both of them is small.
