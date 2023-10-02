# Week 2

## Minibatch Gradient Descent

We use minibatches when the amount of the examples are way too large for the machine to store or compute. To manage with this issue we divide the original massive batch into mini batches. If we have original natch of 5M examples we can divide it into 1000 mini-batches of 1000 examples each.

There is a for loop which iterates between 1 to 1000 for number of mini-batches. And it calculates all the A[l] and Z[l] using that mini-batch. In the end we calculate the cost and perform gradient descent using similar fashion, along all the mini-batches (X{t},Y{t}).

When all the minibatches are run and gradient descent was calculated and parameters were updated, then we call it 1 epoch. 

If we consider batch gradient descent then after every iteration we can see the cost function getting reduced monotomously, but if we use gradient descent then their is noise in the cost function decrement but overall looking globally it reduces monotonously as well. This is because we are using mini-batches which may be giving different distribution for different mini-batches resulting in wobbly cost function decrement. 

We need to calculate the batch size so that the deviation of learning from the cost function is least, and the time taken for the computation is reasonable as well. There is schotastic mini batches in which the mini-batches have size 1 which results in really janky cost function learning, and then there is batch gradient which has a smooth learning path but takes way too long per iteration and is computationally expensive.

We need to find a middle ground for best learning path, which avoids jankiness and takes reasonable time and computational cost, we should also make sure that the GPU and CPU can contain the size of the mini-batches.

## Exponentially weighted averages

Normally when we get data let us consider it theta. Itis mostly all over the place with lot of noise. To mediate this noise and create a smooth graph we use the method of exponentially weighted average. 

Rather than graph of the actual parameter over the time, for e.g. temparature(theta). Rather than looking at example of temperature over time we use exponentially weighted averages given by 

v(t) = (beta \* v(t-1)) + ((1-beta) \* theta(t))

This insures that we combine the previous values with the value of that current day as well which gives us an average between the prior few days. For e.g. if there are 100 days then if beta = 0.9 then, v(t) is averaged over 10 days temperature.

We can see that the graph between v(t) and number of days are exponential. 

We can consider epsilon as 1 - beta. Then if we look at (beta)^(1/1-beta) i.e. (1- epsilon)^(1/epsilon) is equal to 1/e. We inititalise v_t as zero and then we just iterate v_t = beta * v_t + (1-beta) * theta_t, over all the t which is present. For bias correction we can use v_t / (1-beta_t) which corrects the original lagging part of the exponentially averaged graph due to initialization of v_t as zero.

## Gradient Descent using Exponentially Weighted Averages

Rather than W being updated as W = W - alpha \* dW we are are going to update W as W = W - alpha \* v(dW). Where v(dW) = beta \* v(dW) + (1-beta) \* dW.

There are in total 2 hyperparameters alpha and beta. 

We can look as cost function as an ellipse in 3D. The learning can happen due to oscillation of the learning upwards and downwards, and second is towards the minima. We need to decrease the learning upwards and downwards while increasing the speed of learning towards the centre.

We can correlate the learning with motion in real life, where v(dW) = beta * v(dW) + (1-beta) * dW. Where v(dW) being multiplied by beta correlates to velocity, and dW relates to acceleration towards to the minima.

## RMS Prop

We've seen that v(dW) is calculated as v(dW) = beta \* v(dW) + (1-beta) \* dW. But we can calculate this using other method called RMSprop. We calculate another graph which has its points defined by 

s(dW) = beta_2 * s(dW) + (1-beta_2) * d_W^2

There are in total 2 parameters alpha and beta_2.

And the parameters W get updated as W = W - alpha * dW/sqrt(s(dW) + epsilon), here epsilon is a really small value which is there to counter the issue of dividing by 0, it is purposefully extremely small as we dont need it to itself have any imact on learning of parameters. 

## Adam Optimization

Adam Optimization uses best of both worlds i.e. it uses exponentially weighted averages as well as RMSprop and combine everything together for really optimized, smooth and streamlined learning.

We shall consider the case for Weights and the same case is applied to Bias learning as well.

Firstly we calculate, v(dW) = beta_1 * v(dW) + (1-beta_1) * dW

then we correct it with v_corrected(dW) = v(dW) / (1 - beta_1(t))

We simultaneously also calculate, s(dW) = beta_2 * s(dW) + (1 - beta_2) * dW^2

s_corrected(dW) = s(dW) / (1 - beta_2(t))

And then we use both the corrected variables of RMS prop and Exponentially Weighted Average to update the Weight Parameters as 

W = W - alpha * v(dW)/(sqrt(s(dW)) + epsilon)

Adam stands for Adaptive Moment estimation, And the Hyperparameters are alpha, beta_1 , beta_2, epsilon. We need to tune alpha mainly. Values of 0.9, 0.99 and 10^-8 are used for beta_1, beta_2 and epsilon respectively.

## Learning Rate Decay

As the name suggests this is the process of reducing the learning rate after every epoch. This is done because is the learning rate is high then the amount of learning will be great even in the end stages where, we need fine tuning to the global minima of the cost function. 

For this we gradually decrease the learning rate after every epoch with the formula 

alpha = (1 / (1 + (decay_rate * epoch_num))) * alpha_0. Where alpha_0 is the value of learning rate for previous epoch number. 

There are other methods of learning rate decay as well. For example:

1. alpha = (0.95^epoch_number) * alpha_0

2. alpha = k/(sqrt(epoch_num)) * alpha_0

3. We can manually set the value of alpha at different discrete value after every epoch

## Issue of Plateau

We usually deal with cost function which has very high dimensions, in this case the presence of a local optima isn't a problem but a plateau region which gets created in the loss function, this is also called a saddle point. The gradients are extremely small in these regions of the curve which results in less learning and model wastes a lot of time to get out of this region, these kind of regions can be sometimes be avoided using appropriate normalization techniques.


