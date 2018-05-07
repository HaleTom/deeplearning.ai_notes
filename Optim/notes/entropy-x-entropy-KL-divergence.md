# Entropy, Cross-Entropy and KL Divergence


Notes from [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)

Claude Shannon (American Mathematician, Elec engineer, and cryptographer)founded Information Theory in his [1948 paper "A Mathematical Theory of Communication"](http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf).

The goal is to reliably and efficiently transmit a message from sender to recipient.

Not all bits are useful - redundancy, errors.

In Information Theory, to transmit one bit of information is to reduce uncertainty by a factor of $2$.

$
\text{bits communicated} = \log_2(\text{uncertainty reduction factor}) \\[6pt]
\text{uncertainty reduction factor} = \dfrac 1 {\text{probability of event}}
$

So if a message communicated the outcome of an event which was a $0.25$ probability, then the number of bits communicated is:

$\text{bits communicated} = \log_2(8) = 3$

Now, given $\log(\frac 1 x) = -\log x$, we could also say:

$\text{bits communicated} = -\log_2(0.25) = 3 \\[6pt] $

## Entropy

Entropy is a measure of how uncertain the events are.

If there are multiple events which are linked (eg 75% probability sunny, 25% percent chance rain), then the entropy ($H$) is given by:

$\displaystyle H(p) = -\sum_i p_i \log_2 (p_i)$

More generally, it is the average amount of information you get from one sample drawn from a probability distribution, $p$. It says how unpredictable that proability distribution is.

Eg, if living in a desert, the entropy about rain / not rain will be close to 0.


## Cross-Entropy

Cross-entropy is the average message length.  Ie, the sum of:

Message length x probability of that message being sent

We want the most common messages to be composed of the minimum number of bits while still having them be distinguishable.

For a 2-bit message, optimally this message would be sent with a probability of $0.25$, or $\frac 1 {2^2}$.

Given:

$p$ = the true probability distribution
$q$ = the predicted probability distribution

We get the formula for cross-entropy:

$\displaystyle H(p,q) = -\sum_i p_i \log_2 (q_i)$

The amount by which the cross-entropy exceeds the entropy is called the relative entropy, or the Kullback–Leibler divergence, KL divergence.

Cross Entropy = Entropy + KL divergence

## In machine learning

In a classifier, we use the cross entropy between the ground truth probability distribution $y$ (one-hot, all $0$s except for one $1$), and the predicted probability distribution, $\hat y$.

We use the natural logaritm, $\ln$ rather than $\log_2$. 

$\log_2(x) = \dfrac {\log_e x} {\log_e 2}$

Ie, take the natural logarithm and divide by a constant, $\ln 2$.

In the one-hot case, all the $0$s in $p_i$ add nothing to the overall sum, and the result is given only by the case where the ground truth value is $1$.
