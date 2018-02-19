# Week 2

Logisitic regression is an algorithm for binary classifiction: detecting the presence vs absence of something.

For each red, green, blue matrix representing the image, concatenate all values to get a feature vector.

$n_x$ = the dimension of the vector $x$. Sometimes just $n$.

A single training example is represented by: $(x,y)$ where: $x \in \mathbb R^{n_x}$ and $y \in {0,1}$

$m$ = number of training example, the $i$-th represented as $(x^{(i)}, y^{(i)})$

The training set matrix, $X$, is organised with the columns being the individual training examples.  $X \in \mathbb R^{(n_x, m)}$
This organisation will make the implementation much easier.

    X.shape == (n, m)

$Y$ is also organised with the examples in columns.

## Logistic Regression

Logistic regression can be seen as a very small neural network.

$y$ is the "ground truth" label, given from the training set.
$\hat y$ is the predicted value of $y$ from the network.

Given $x$ get $\hat y = P(y=1|x)$

Paremeters: $w \in \mathbb R^{n_x}, b$

For easier implementation, we keeep $w$ and $b$ separate. $b$ is the bias or intercept term. 

In the previous course, $w_0$ was equal to $b$ and $x_0$ was always 1 ($x \in \mathbb R^{n_x+1}$).

$\hat y \ne z = w^Tx + b$ as this can be negative, or greater than one (this would be linear regression)

Instead, we use $\hat y = \sigma(z)$ to map the range to $(0,1)$.

$\sigma(z) = \frac{1}{1 + e^-z}$

Objective: find $w$ and $b$ such that $\hat y \approx y$, ie, $\hat y$ is a good estimate of the probabilty that $y = 1$

### Loss (Error) / Cost function

To train $w$ and $b$, we first need to define the loss function.

Logistic Regression doesn't use the squared error as this can have more than one minimum.  Instead we use:

$
L(\hat y,y) = - \Big(y\log \hat y + (1 - y)\log (1 -\hat y)\Big) \\[6pt]
$

Terminology: The *loss* functions is applied to only a single example, whereas the *cost* function is the is applied to the parameters.

Cost function:

$$\begin{align*}
J(w,b) &= \displaystyle \frac 1 m \sum_{i=1}^{m}{L(\hat y^{(i)},y^{(i)})} \\
    &= -\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}\log\hat y^{(i)} + (1-y^{(i)})\log(1-\hat y^{(i)})\right)
\end{align*}$$

## Gradient Descent

We want to find values of $w$ and $b$ which minimise $J(w,b)$.

One of the reasons we chose the cost function as we do is because it is guaranteed to be convex, so all negative gradients point at the minimum.

Until convergence (within a $\epsilon$ value), we repeatedly change each element $w_i$ to be:

$\displaystyle w_i' := w_i - \alpha \frac {\delta J(w_i,b)} {\delta \ w_i}$

Where $\alpha$ is the learning rate, or size of the step towards the minimum. (The analogous update is done for $b$.)

$\delta$ is used to represent the partial derivitive (the term on the top depends on more variables than the one given on the bottom).

Convention: `dw` is used to represent the derivitive term

## Derivitives

* In python, instead of writing `dFinalOutputVariable_dVar` we just write `dvar`.
* [Derivative Calculator](https://www.derivative-calculator.net/) provides worked examples and rules
* [Wolfram Alpha](https://www.wolframalpha.com/) can calculate derivatives (but doesn't simplify sigmoid fully).
* [Fully worked example of derivatives of logistic regression loss function](http://ronny.rest/blog/post_2017_08_12_logistic_regression_derivative/)

Given $a=\sigma(z)$, then the [derivative of the sigmoid function](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x) is:

$$\frac{da}{dz} = a(1-a)$$

[More friendly sigmoid derivation](http://ronny.rest/blog/post_2017_08_10_sigmoid/)

Given the loss function (from above):
$$ L(a,y) = - \Big(y\log a + (1 - y)\log (1 -a)\Big) $$

The derivative is:

$$\frac{dL(a, y)}{da} = - \frac y a + \frac{1-y}{1-a}$$

Multiplying these two with the chain rule, we get:

$$ \frac{dL}{dz} = \frac{dL}{da} \cdot \ \frac{da}{dz} = a-y $$


------

Given the cost function (from above):

$$\begin{align*}
J(w,b) &= \displaystyle \frac 1 m \sum_{i=1}^{m}{L(\hat y^{(i)},y^{(i)})}

  \\
    &= -\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)})\right)
\end{align*}$$

[The derivative of the cost function is](https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated):
$$ \frac{1}{m}\sum_{i=1}^m\left[h_\theta\left(x^{(i)}\right)-y^{(i)}\right]\,x_j^{(i)} $$

![logistic regression derivatives](wk2-logistic-regression-derivatives.png)

## Logistic regression with $m$ training examples

![logistic-regression-on-m-examples](wk2-logistic-regression-on-m-examples.png)

* This slide implements a single step of gradient descent. It will need to be run multiple times until convergence.
* The two `for` loops marked in green should be vectorised for efficiency (especially important with large training sets)


---------------

## Other

[Precision vs recall with venn diagrams](http://ronny.rest/blog/post_2018_01_26_precision_recall/)

[Swish function like ReLU but differentiable at all points](https://www.derivative-calculator.net/#expr=x%2A%281%2F%281%2Be%5E-x%29)

## TODO

[Yes you should understand backprop - Andrej Karpathy](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
