
# Heros of Deep Learning

## Geoffrey Hinton

1987 Hinton, McLeland? Recirculation algorithm

Generative Adversarial Nets are currently a new idea

Advice for breaking into DL:

Read the literature, but don't read too much of it.

For creativity, read enough to notice something you believe everything is doing wrong, where it just doesn't feel right, and then figure out how to do it right. And when people tell you that's no good, just keep at it. 

Hinton's principle for helping people keep at it:

> Either your intuitions are good or they're not. If you intuitions are good, you should follow them and you'll eventually be successful. If your intuitions are not good, it doesn't matter what you do. You might as well trust your intuitions. There's no point not trusting them.

> In summary, read enough to develop your intuitions, and then trust your intuitions. Go for it. Don't be too worried if everyone else says it's nonsense.

> If you think it's a really good idea and other people tell you it's complete nonsense, then you know you're really onto something. ... That's the sign of a really good idea.

Ng recommends people replicate results in papers. Hinton says this ensures you know all the little trick required to make it work.

Hinton's other advice: Never stop programming. Work around any issues that come up.

See if you can find an advisor who has beliefs and interests similar to yours. They they will give a lot of good advice and time. 

Right now there aren't enough academics to train the people needed in industry. 

Our relationship to computers has changed. Instead of programming them, we show them and they figure it out. Most faculties don't have anything near a 50-50 balance.

Google is training brain residents. 

### Paradigms for AI:

1950s: Von Neumann and Turing didn't believe in symbolic AI. They were far more inspried by the brain, but they died too young and their voice wasn't heard. In the early days, people were convinced that the representations needed for intelligence were symbolic expressions, or cleaned up logic and that intelligence was reasoning.

Now the view is that a thought is a vector of neural activity, rather than a symbolic expression. Words come in, words go out, so people assumed that what was in the middle was a string of words/symbols.  Hinton belives that what's inbetween is nothing like that. Big vectors cause other big vectors, and that's totally unlike the view that thoughts are symbolic expressions.

Some of AI is coming around to this viewpoint, but too slowly in Hinton's opinion.

## Pieter Abbeel - ML, DL and Robotics
Studied electrical engineering. Everything was interesting, he didn't know where to start. AI seemed to be at the core of everything and could assist any discipline so he picked that.

Deep reinforcement learning: Started with reinforcement learning. Autonomous helo flight, folding laundry. Learning enabling what would not be possible otherwise. Domian expertise and ML expertise was required. Deep learning was able to represent the domain expertise.

Was inspred by Alexnet's 2012 success to apply deep learning to reinforcement learning. 

What's next: 
* Where does the data come from?
* Credit assignment - what actions done early on got the credit later on?
* Safety problem - autonomously collected data would be very dangerous on public roads, yet negative examples are needed

The deep part is the representation of the pattern, which has been largly addressed. How to tease apart the pattern is still a big challenge.

How to get systems to reasons of long time horizons. Currently problems are where acting well over 5 seconds means that the entire problem is well addressed. This is very different from a day-long skill or living a life a s a robot.

Also how to keep learning when you're already pretty good. Catching the one in a billion accident data is really important.

Can we learn the reinforcement learning model itself (credit assignment, exploration)?  Have another learning program be the outer loop, tweaking parameters which make the learning of the actual task faster.

Rather than starting from scratch, how can we reuse what's already learned for the next task?

Pieter appreciated Andrew Ng's supervision which focussed his Ph.D on the real-world impact rather than on the math itself.

Advice: don't just read stuff, but try it out yourself. Do some Kaggle.

PhD or job in big company? Depends on the amount of mentoring you can get. The one or even two advisors have mentoring as their defined role, and often love guiding and shaping students. Companies may do this, but it is not the crux of the dynamic.

DeepRL successes:
* Playing Atari games based on pixels alone
* Simulated robot inventing walking / running rewarded by linear distance and minimal ground impact

Behavioural cloning: Supervised learning to mimic what a human is doing, then gradually layering on reinforcement learning to better meet metrics.

RL is fascinating to watch the learning of what works, but is time consuming and not always safe.

#### TODO based on Pieter's recommendations:
* Andrej Karpathy's deep learning course [Convolutional Neural Networks for Visual Recognition (CS231n)](http://cs231n.stanford.edu/)
* Berkley's deep reinforcement learning course

-----------

# Ruslan Salakhutdinov

Director of Research at Apple, Prof at Carnegie Mellon.

Masters at Toronto, then took a year in financial sector. Bumped into Hinton who invited him to his office and inspired him.  Started PhD with Geoff 3 months later.

Co-authored one of the early papers on RBMs.

Progress moved from RBMs to Deep RBMs. 

What's happening with RBMs?

Around 2009/2010 we realised that with GPUs coming on to the scene, training a deep NN was giving the same or better results than pre-trained or stacked RBMs. BMs and RBMs were previously needed to initialise NN weights before the greater compute required for backprop.

VAEs, RBMs, GANs are generative models. Learning algorithms require Markov Chain Monte Carlo, Variational learning, etc which are not as scalable as backprop. More efficient ways of training these models are needed.  Convolution is difficult to integrate.

Variational Auto Encoders can be looked at as directed versions of Boltzmann Machines.  Reparameterisation trick allows back prop within a stochastic system. But we haven't yet worked out how to do something similar with RBMs.

Generative semi-supervised, and unsupervised vs supervised approaches:

Most of the successes recently have been in supervised learning. 

In the early days, unsupervised learning was viewed as unsupervised pre-training as we didn't know how to train multi-layer systems.  Even today, working with lots of unlabelled data and few labelled examples, unsupervised pre-training or building generative models can help with supervised tasks.

GANs, VAEs. His lab is currently working on Deep Energy Models. 

Semi-supervised learning is the next frontier. Where some hints are given from some examples and then lots of unlabelled data is given to annotate.

Advice for people wanting to enter research or applied work:

Try different things and not be afraid to try new things. Don't be afraid to tackle problems that others say are impossible.

Implement the algorithms in code at the low level.

PhD vs Company
Apple has a mix of students who want to take both routes. 

Academia: More freedom to work on long-term problems. Or some odd-ball problem. More freedom.

Industry: Very exciting as the research is geared toward what will impact millions of users.  More resources in terms of compute. 

There is lots of cross-pollination between the two. 

Frontiers for research:

* Unsupervised learning
* Deep reinforcement learning - training agents in virtual worlds. Getting agents to communicate with each other. 
* Reasoning and natural language understanding
* Being able to learn from fewer examples

--------

# Yann LeCun

Was always interested in intelligence and human evolution, space and technology.

Studied electrical engineering. Stumbled on a book by Noam Chomsky and Jean Piaget (child development psychologist), a debate on nature vs nurture. Seymour Papert from MIT (on Jean Piaget's side), had worked on the Perceptron. Yann found everything he could on machine learning, but found that development stopped in the 1960s (this was roughly 1980). He did some projects with mathematics professors but the field had disappeared.  He read about neuroscience.

On finishing engineering, he studied chip design.  The unsolved question from the 1960s was how to train a neural network with multiple layers. He read about hierarchy and [Fukishima's article on the neocognitron](http://vision.stanford.edu/teaching/cs131_fall1415/lectures/Fukushima1988.pdf) which is similar to what we now call CNNs but without backprop.  A society gave him a paper on Hopfield networks, or associative memory with neural nets. This paper renewed interest in the 1980s by physicists and psychologists.  ["Optimal perceptual inference"](https://papers.cnl.salk.edu/PDFs/Optimal%20Perceptual%20Inference%201983-646.pdf) - the first paper on Boltzmann machines by Hinton and Terry Sejnowski This paper talked about hidden units, learning multi-layer NNs more capable than linear classifiers.  He met Terry in 1985 and talked about backprop he was working on at the time. Terry told Hinton, and Hinton asked for him when Hinton was in a speaker in France.  Backprop was really invented in the context of Optimal Control in the 1960s (the Adjoint State method).  Back prop was popularised by Rumelhart Hinton Williams [Learning representations by back-propagating errors](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)

In AT&T, Yann invented LeNet.  He started working on shared weight (convolutional) NNs as a postdoc under Geoffrey Hinton. In Bell Labs he scaled up the network and had a Sun 4 all to himself (vs to a department at university).  He worked on the USPS training set (5000 examples) and got better results than anyone else using CNN methods.  There was convolution with strides, but no subsampling or pooling layers.  Each convolution was subsampling directly - they couldn't afford the computation to have convolution at every location. The 2nd iteration had convolution, pooling layers and subsampling and was called LeNet1.

LeNet1 allowed for thing like cheque reading machines from AT&T subsidiary NCR (National Cash Register). In 1996, AT&T split into AT&T, NCR, and Lucent. The patent went to NCR but the engineering teams were in Lucent, so not much further progress could be made (the patent expried in 2007).

The LeNet paper was partially about the neural network, and partially about automata machinery. Few have read the 2nd part about sequence level discriminatory learning and structure prediction without normalisation. This is similar to what we know as CRF (Conditional Random Field) today. 

[Difference between pooling and subsampling](https://stats.stackexchange.com/q/354944/162527)

As department head, Yann started DejaVu which was to compress scanned documents for the newly booming internet. 

Dark period between 1995 and 2002 when not many people were working on Conv Nets. 

ImageNet 2012 Alex talked about Convnets like everyone should know what they are.... and now they do.

Yann's view on corporate research: 

Was given authority to setup FAIR (Facebook AI Research) as Yann wanted.  He ensured:

* Research needs to be open
* Researchers need to be encouraged or even mandated to publish and evaluated like academic publishing

### Advice for people wanting to break into AI

Make yourself useful. 
* Make a contribution to an open source project
* Make an implementation of an algorithm whose code doesn't exist online
