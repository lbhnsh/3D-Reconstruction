# Course 3: Structuring Machine Learning Projects 

## Week 1: 

### Why ML Strategy? 
Suppose the NN you built (say cat classifier) has 90% efficiency which isn't practically good, so you might spend a lot of time in increasing it by getting more data but it may not work and would release in a lot of time wastage, thus ML strategy is required. 

### Orthogonalization
Orthogonalization is just tuning one factor which you are sure of and tuning which would increase efficiency. 

#### Chain of Assumptions in ML
1. Fit training set well on Cost function. 
2. Fit dev set well on Cost function. 
3. Fit test set well on Cost function. 
4. Performs well in real world.

### Single Number Evaluation Metric

Precision and Recall

One Classifier may have high precision but low recall
Another may have high recall but low precision

So we combine the metric as harmonic mean of precision and recall

F1 sum= 2/(1/P+1/R) 
 
### Satisficing and Optimizing Metric

The metric that has to be maximum (optimum) is the Optimizing Metric
The metric that has only a sufficing condition is the Satisficing Metric


When there are N metrics:
1 Optimizing Metric
N-1 Satisficing Metrics

### Train/ Dev/ Test Distributions

**Let Dev Test and Test set from the same distribution**

CHOOSE A DEV SET AND TEST SET TO REFLECT DATA YOU EXPECT TO GET IN THE FUTURE AND CONSIDER IMPORTANT TO DO WELL ON.


### Size of Dev and Test Sets

Old way of split: 60 20 20 : train dev test

**Set your test set to be big enough to give high confidence in the overall performance of your system**

### When to Change Dev/ Test Sets and Metrics?

Error= (1/sum of weights )*sum(loss(weight(i)( y pred + y ))


If metric and/or dev/test set does not do well change metric and/or dev/test set.
They may work better on a different set or metric.


### Why Human-level Performance?
 
Bayes Optimal Error: Best possible error, no model can reduce error than this limit

As long as ML is worse than humans:
    - Get labeled data from humans
    - Gain insights from manual error analysis
    - Better analysis of bias and variance

### Avoidable Bias

Consider human error as more than Bayes error:
Then room is to reduce bias

Consider human error as Bayes error:
Then room is to reduce variance ( error percent between dev and training set)

Avoidable bias: percent of bias difference between bayes error and training error (that is acceptable as it is small.)

### Improving your Model Performance

Avoidable bias:
1. Train bigger model
2. Train better/longer optimization algorithms
3. NN architecture/hyperparameters search

Variance:
1. More data
2. Regularization
3. NN architecture/hyperparameters search



## Week 2:

### Carrying Out Error Analysis
Error analysis:

if its a cat classifer and error is more on dev set of dog images
1. Get ~100 MISLABELED dev set examples
2. Count up how many are non wanted images (dog images)

Ceiling of performance : We find how many percent of that non-cat images are there

say there are 5/100 dog images error would max reduced to 5% of the existing error but if there are
50/100 dog images error would reduce by 50% of existing error

So we can decide to spend time on what type of images (say here dog images)

#### Evaluate multiple ideas in parallel

Table with columns of classification of misjudged data and one of comments

### Cleaning Up Incorrectly Labeled Data

(When default human data is wrong for some case)

Deep Learning Algorithms are ROBUST to random errors.

If there is a pattern in the error training will be wrongly done.

#### Correcting incorrect dev/test set

1. Apply same process to dev/test to make sure they come from same distribution
1. Consider examining examples your algorithm got right as well as ones it got wrong
3. Train and dev/tes data may now come from slightly different distributions

### Build your First System Quickly, then Iterate
First build a simple model, assess errors then move forward to repair them

### Bias & Variance with Mismatched Data Distributions
If training and dev sets are from different distributions we can't really talk about bias and variance

New term: training dev set
**Training dev set** part of training set which is used as dev set and not for training

Consider errors in each levels and the gap between each layer denotes which issue

1. Human level - Training set -> Avoidable bias
2. Training set - Training dev set -> Variance
3. Training dev set - Dev set -> Data Mismatch
4. Dev set - Test set -> Degree of overfitting to devset

### Addressing Data Mismatch
1. Carry out manual error analysis to try to understand difference between training and dev/test sets
2. Collect more dara similar to dev/test sets
#### Artificial data synthesis
Speech synthesis, car images,etc. refer video



### Transfer Learning
Learn for a different task and apply it for other task.

Training on a previous different set is called **Pre-Training**

Transfer Learning is useful when there is lot of data for pre-training and less data for actual task.
*Opposite case doesnt work for transfer learning*

### Multi-task Learning

Even when some labels are '?' summation is omitted over such terms in a matrix 

When does it work?
Usually when amount of data you have for each task is quite similar

For it to work neural network should be big.

### End-to-end Deep Learning

if there is a large amount of data, the whole pipeline (broken into smaller tasks) is not needed, AI trains itself and maps x to y

### Wheter to use End-to-end Deep Learning
Pros:
1. Let the data speak
2. Less hand-designing of components needed

Cons:
1. May need large amount of data
2. Excludes potentially useful hand-designed components


Key question: 
**Do you have sufficient data to learn a function of the complexity needed to map x to y?**




