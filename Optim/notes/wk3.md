# [Week 3 - Hyperparameter tuning, Batch Normalization and Programming Frameworks](https://www.coursera.org/learn/deep-neural-network/lecture/dknSn/tuning-process)

* [Forum](https://www.coursera.org/learn/deep-neural-network/discussions/weeks/3)
* [mx's notes](http://x-wei.github.io/Ng_DLMooc_c2wk3.html)

## Hyperparameter tuning process

How to systematically organise the hyperparameter tuning process.

The hyperparameters determine the final values of each layer's $W$ and $b$ that are learned. There are many:

 * Learning rate ($\alpha$)
 * Mini-batch size
 * Number of iterations
 * Activation functions ($g^{[1]}, g^{[2]}, ...$)
 * Regularisation parameters
 * Number of hidden layers ($L$)
 * Number of units per hidden layer ($n^{[1]}, n^{[2]}, ...$)
 * Learning rate decay
 * Momentum term ($\beta$)
 * Adam terms ($\beta_1, \beta_2, \varepsilon$)

 Andrew recommeds the following order of tuning:

### First tune
The learning rate, $\alpha$ is the most important hyperparameter to tune. 

### Second tune
After $\alpha$, Andrew looks at:

 * Momentum term ($\beta_1$)
 * Mini-batch size
 * Number of hidden layers ($L$)

### Third tune
 * Number of hidden layers ($L$)
 * Learning rate decay

### Fourth tune
 * Adam default parameters - Andrew doesn't bother tuning these.

### Searching the parameter space

Don't use a $n$ x $n$ grid. Instead use $n^2$ random values:

![wk3-grid-vs-random.png](wk3-grid-vs-random.png)

* Grid search: only $n$ distinct values are tried in one dimension
* Random choice: can have $n^2$ distinct values per dimension

### Coarse to fine

Zoom in to smaller performant regions of hyperparam space and re-sample more densely.

![wk3-coarse-to-fine.png](wk3-coarse-to-fine.png)

## Using an appropriate scale for picking Hyperparameters

Sample at random, but at appropriate scale, not uniformly.

Sampling uniformly at log scale is more resonable as there are equal samples taken from each part of the scale.

This method is not so important when using coarse to fine as that method will home in on values selected by the poorer selection of uniform search space.

Example: choice of alpha in $[0.001, 1]$:

![wk3-search-scale.png](wk3-search-scale.png)

In general, when searching for $\alpha$ between $10^a$ and $10^b$, then set $r \in [a, b]$, and then set $\alpha = 10^r$.

In the above example:

    r = -4 * np.random.rand()  # -4 < r <= 0, uniformly at random
    alpha = np.exp(10, r) # 10e-4 < alpha <= 1.0

### Picking $\beta$ for exponentially weighted averages

Instead of searching in $[0.9, 0.999]$, subtract from $1$ and search $r \in [0.1, 0.001]$:

$ \begin{alignat}{1}
r &\in [-3, -1] \\
1 - \beta &= 10^r \\
\beta &= 1 - 10^r \\
\end{alignat}$

It's important to do this as the formula for the approx number of datapoints averaged over, namely $\dfrac 1 {1- \beta}$ is very sensitive in changes when $\beta$ is close to $1$.

Eg, when searching $\beta \in [0.9000, 0.9005]$, both ends of the range average over $\approx 10$ datapoints.
However, searching $\beta \in [0.999, 0.9995]$ will average over $1000$ at the low end to $2000$ at the high end.

The above technique will sample more intensely in the regime when $\beta$ is close to $1$ (or when $1 - \beta$ is clost to $0$).

## Hyperparam tuning in practice: Pandas vs Caviar

Deep learning is applied to many different domains and intuitions about hyperparams from one domain may not transfer to another.

Hyperparam choice can get stale, even in the same domain. The algorithm, data or compute available may change and it's good to re-evaluate at least once every several months.

In terms of hyperparam search, there are two schools of thought:

### Panda
* Babysit a single model, tuning parameters daily, weekly as learning progresses
* Usually done with a huge dataset but not a lot of compute
* When one can only afford to train one model, or a very small number

### Caviar
* Train many models in parallel
* Usually done when there is more compute available and the dataset isn't too huge
* Pick the one which performs the best

## Batch normalisation

Batch normalisation was developed by Sergey Ioffe, Christian Szegedy in [this paper](https://arxiv.org/abs/1502.03167).

It makes hyperparameter search much easier and the network more robust to the choice of hyperparameters (a larger range will work well).

Previously, we saw how to normalise the input layer:

$ \begin{alignat}{1} \displaystyle
\mu &= \dfrac 1 m \sum_{i=1}^m z^{(i)} \\[6pt]
\sigma^2 &= \dfrac 1 m \sum_{i=1}^m \left(z^{(i)} - \mu \right)^2 \quad \text{# element-wise for matrix} \\[6pt]
\hat z^{(i)} &:= \dfrac {z^{(i)} - \mu } {\sqrt{\sigma^2 + \varepsilon}} \\[6pt]
\end{alignat}$

The same process can be applied to the output of any hidden layer. Normalisation can be done on either $z$ or $a$, but is much more often done on $z$.

The above gives mean $\mu = 0$ and variance $\sigma^2 = 1$. To tune the mean and variance, we use:

$\tilde z^{(i)} = \gamma \hat z^{(i)} + \beta $

Where $\gamma$ and $\beta$ are learnable parameters of the network, updated by gradient descent just like the $W$s and $b$s.

If $\gamma = \sqrt{\sigma^2 + \varepsilon}$ and $\beta = \mu$ this inverts the normalisation so that $\tilde z^{(i)} = z^{(i)}$.

As input to the activation function, use $\tilde z^{(i)}$ instead of $z^{(i)}$.




|  Date      | Name                                                | Author(s) |
|------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------
| 2015-02-11 | [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)| Sergey Ioffe, Christian Szegedy
