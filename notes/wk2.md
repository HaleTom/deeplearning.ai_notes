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
J(w,b) &= \displaystyle \frac 1 m \sum_{i=1}^{m}{L(\hat y^{(i)},y^{(i)})}

  \\
    &= -\frac{1}{m} \left(y^{(i)}\log\hat y^{(i)} + (1-y^{(i)})\log(1-\hat y^{(i)})\right)
\end{align*}$$

## Gradient Descent
