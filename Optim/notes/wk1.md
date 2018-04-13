# [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome)

This course will cover:
* Hyperparameter tuning
* Data setup
* Ensure optimisation algoritm runs quickly

# Week 1 - Practical aspects of Deep Learning
This week:

* Setup of machine learning problems
* Regularisation
* Tricks for ensuring the implmentation is correct


## Applied ML is a higlhly iterative process

Intuitions from one ML domain often don't translate well into a new domain.

Decisions require are:

* Number of layers
* Number of hidden units
* Learning rate
* Activation functions
* ...

Use the Idea -> Code -> Experiment cycle

The rate of progress is determined by how quickly we can iterate this cycle.

Setting up the data well can make this much more efficient.

## Train / Dev / Test sets

To allow:
* Faster iteration of the Idea -> Code -> Experiment cycle
* Measurement of bias and variance of the algorithm

Split the data into:
* Training set
* Hold out cross validation set / "development set"
* Test set

The models are evaluated on the dev set and then the best one is tested against the test set for an unbiased estimate of performance.

Previously data was split 70% train / 30% dev  or 60% train / 20% dev / 20% test. This was consiered best practice in machine learning. These ratios still hold for $ 100 < m < 10,000$ examples.

In the modern big data area (eg 1,000,000 data points) the dev and test sets have become a much smaller percentage of the total. The dev set just needs to be large enough to evaluate different algorithm choices. 10,000 examples may be sufficient. Likewise with the test set.

Andrew also mentions a 99.5% / 0.4% / 0.1% split as possible.

More specific guidelines will come later on in this Coursera specialisation.

Not having a test set might be ok if no unbiased estimate of performance is needed. In this case the dev set is generally called the test set, even though it's being used as a hold-out cross validation set and the terminology is confusing. 

### Mismatched train / dev / test distributions

More and more people are using differing train / test data distributions.

Allowing training set augmentation can increase overall performance.

Eg, the training set could be pictures from commercial galleries (eg more professional pictures), and the dev and test sets could be from user uploads (happy snaps).

Rule of thumb: Make sure that the dev and test sets come from the same distribution.

## Bias and Variance

Most of the really good machine learning practitioners have a really good understanding of bias and variance. It's easy to learn but difficult to master - there is a lot of nuance.

The bias / variance trade-off is less of a trade-off in the deep learning area. Understanding the individual terms is still important.

![wk1-bias-variance.png](wk1-bias-variance.png)

In higher dimensions where decision boundaries can't be visualised, we use the combination of train and dev set errors.

Error is relative to the Bayes or optimal error.

* **High bias / underfitting** - an over simplistic idea of the function which fits the data
 * If the training set error is high then there is high bias. If the dev set error is even higher then there is high variance as well.

* **High variance / overfitting** - a too complex function which fits the training set too well, making the dev set error higher
 * If only the dev set error is high, then there is high variance.

 All the above assumes that the Train and Dev sets are drawn from the same distribution. Otherwise there is a more sophisticated analysis coming later in the course.

 ![wk1-high-bias-and-high-variance.png](wk1-high-bias-and-high-variance.png)

Here a function with both high bias and high variance has it's decision boundary shown in purple, against training set data with outliers.

With high-dimensional inputs, the example doesn't need to be so contrived - it's more common.

## Basic Recipie for Machine Learning

1. Does the model have high bias (on training set performance)?
 * Bigger network (almost always helps)
 * Train longer (never hurts)
 * Different NN architecture (may help, harder to be systematic in the approch)
 * Try advanced optimisation algorithms
1. Does the model have high variance (ie, increased error on dev set)? 
 * Use regularisation
 * Get more data (often expensive, time consuming)
 * Different NN architecture (may help, harder to be systematic in the approch)

In the pre-deep learning area, bias / variance trade-off meant that decreasing one increased the other.

With deep learning, as long as one can:
 * train a bigger network
 * get more data

then one can be decreased without any increase in the other. This is one reason that deep learning has been so effective for supervised learning.

Using regularisation will increase the bias a little bit, but not too much if the network is large enough. 

## Regularisation

Generally the first thing to try with a high variance problem is to use regularisation. 

Getting more training data is better, but it can be expensive and time consuming.

L2 regularisation is used much much more often.

### L2 regularisation (logistic regression case)

We take the cost function from before and add a new term:

$$\begin{align*}
J(w,b) &= \displaystyle \frac 1 m \sum_{i=1}^{m}{\mathcal L(\hat y^{(i)},y^{(i)})} \\
J^+(w,b) &= \displaystyle \frac 1 m \sum_{i=1}^{m}{\mathcal L(\hat y^{(i)},y^{(i)})} + \frac{\lambda}{2m} \left\Vert w \right\Vert_2^2
\end{align*}$$

Where:
$$ \left\Vert w \right\Vert_2^2 = \sum_{j=1}^{n_x} w_j^2 = w^Tw $$

and $\lambda$ is the regularisation hyperparameter.  (Note `lambda` is a reserved word in Python, so use `lambd` instead)

$w^Tw$ is the squared Euclidean norm of the vector $w$.  This is also called the $L2$ regularisation.

$b$ can also be regularised, but in practice it makes very little difference. Most of the parameters are in $w$ rather than in $b$.

### L1 regularisation (logistic regression case)
Instead, this term can be added:

$$ \left\Vert w \right\Vert_1 = \sum_{j=1}^{n_x} \left|w_j\right| $$

With this method, $w$ will end up being sparse, meaning there are zeros in the vector. This is because the of the additional term gradient doesn't decrease. [More info here](https://stats.stackexchange.com/a/159379/162527) or [secondarily, here](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/).

### Regularising Neural Networks

For a matrix, it's not called the L2 norm, rather the Frobenius norm, denoted:

$$ \left\Vert W \right\Vert_F^2 = \sum_{i=1}^{n^{[l]}} \sum_{j=1}^{n^{[l-1]}} (W_{i,j}^2) $$

This is summed over all layers of the network.

**Backward propagation**

Before we had:
$$ dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}$$

Now we add the term:

$$ dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} + \frac \lambda m W^{[l]}$$

The update is the same as before:

$$ W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]} $$

L2 regularisation is also called "weight decay". This is because $W^{[l]}$ is multiplied by by the less-than-one term $(1 - \frac{\alpha \lambda}{m})$ after factorising for $W^{[l]}$.

## Why does regularisation prevent overfitting?

If $\lambda$ is large, then $W^{[l]}$ values will need to be small to keep the overall cost low.

If $W$ is small, then $z = W^{[l]}a^{[l-1]}$ will also be small. With small values of $z$ and the tanh activation funciton, the output of the tanh will be in the linear section, simplifying the layer to be more like a linear logistic regression unit. Many sequential linear units can be calcuated by a single linear unit. Keeping things more linear reeduces the complexity of the function that the network can calculate, and making for a simpler decision boundary, and less overfitting.




1c 2a 3d 4b 5b 6a 7a 8bc 9abdf 0c

