# Week 1 - [Yoshua Bengio](https://www.coursera.org/learn/deep-neural-network/lecture/bqUgf/yoshua-bengio-interview)


One of the icons of deep learning. 


Read a lot of SciFi. Read a lot of papers in 1980s. Mentioned Hinton as a connnectionist. 

Worked on Recurrent Nets, speech recognition, HMM (graphical models). 

Discovered issues with long-term dependencies with training NNs. 

How has his thinking evolved over time?

We start with experiments and intuition. Theory comes later. We now understand why backprop is working so well and why depth is so important, but there was no proof for these in the beginning. 

Biggest mistake: thinking like everyone else in the 90s that smooth non-linearities were required for backprop to work.

What is the relationship between deep learning and the brain?  Connectionists say that information is distributed across the activation of many neurons, rather than by a single "grandmother cell" or symbolic representation as classical AI would say. 

Built some relatively shallow but very distributed representations for word embeddings. 

Most proud of long-term dependencies, working on the curse of dimensionality, stacks of autoencoders / RBMs, initialisation to prevent vanishing gradients. With unsupervised learning: denoising encoders, GANs, translation using attention.

Attention changed Yoshua's view on NNs from something that can map a vector to a vector. With attention mechanisms, any type of data structure can be handled. 

In relation to biology, what is like backprop, but could actually be implemented in a brain? Is there a more general principle at play behind backprop? Is there a better way to do credit assignment? 

Unsupervised learning is really important. Supervised learning requires humans to define what the important concepts are and label them. But humans discover new concepts by interactions with the world, eg a 2-year-old understanding physics.

He's been combining unsupervised learning with reinforcement learning. In the beginning (about 2000), RBMs and autoencoders were focussed on learning good representations (which is still a central problem). But what is a good representation. What's a good objective function which will tell us? [A meta objective function?] 

What in deep learning excites him the most?

Direction of research going back to principles, having an AI observe the world, interact with the world and discover how that world works.  Even if it's as simple as a video game. How do we deal with new domains (transfer learning) or categories where there are too few examples (generalisation)? 

A lot of the successes in AI have been with perception. What's left is high-level cognition - understanding at the abstract how things work. Reasoning, sequential processing, causality and how a system can work this out autonomously. 

He's a fan of simple problems which can be understood more easily. The research cycle can also be must faster. 

Likes math but says it's more important to be able to convince people logically. Math allows us to make that theory stronger and tighter. 

"What kind of question should we ask that would allow us to understand better the phenomena of interest?" Eg what makes training of recurrent nets so much harder?

He's a fan of experiments not to improve an algorithm, but to better understand the algorithms we currently have, and why they work. He emphasised the why again as the goal of science. 

Advice for people getting into AI:

The depth of understanding of a researcher is different to an engineer using ML to build products. 

In both cases: Practice. Program the things yourself to ensure you really know what's going on. 

Try to derive things from first principles. 

Reading, looking at code, writing code, make sure that you understand everything you do. Especially for the science part of it ask: "Why are people doing this?"

Wrote [Deep Learning](http://www.deeplearningbook.org/) (the book with too much maths) with Ian Goodfellow and Aaron Courville.

To keep abreast: International Conference on Learning Representations (ICLR) proceedings were recommended for really high quality papers. Also Neural Information Processing Systems (NIPS) and nternational Conference on Machine Learning (ICML).

How to become good at deep learning?

Develop the intuitions and don't be afraid of the math. The math will become clear when you understand what's going on. With a backgroun in CS and math, he says in about 6 months one can learn enough to start research experiments. Probability, algebra, calculus and optimisation were mentioned as the add-ons to CS.


# Yuanqing Lin

Head of Baidu research. Selected by Chinese goverment to build national deep learning research lab.

The lab's mission is to build hopefully the biggest deep learning platform, offering compute through Paddle Paddle, massive data sets, and uptake in Baidu. 

The lab will allow someone to publish code, dataset and compute configuration so that results can be easily reproduced, within a few seconds.

Majored in optics. Very good math background. Did PhD at UPenn? university, wanting to do something new.

Started working on computer vision relatively late. Won the first ever of Imagenet challenge. 

## What should people outside of China know about DL in China?

DL is really boming. Search enginines, face recoginition, surveillance, e-commerce. 

There is a positive feedback loop which starts with a toy dataset and a good enough algorithm, which then gets applied to real-world problems and gets access to more data to produce and even better algorithm, which gets excellent performance and then access to even larger amounts of data. He drew an exponential curve in the air at 9:56 :)

He really only investigates R&D opportunities which have this positive feedback loop as other companies will likely be better at linear development.

## What advice for someone entering AI / DL?

Start with open source frameworks, TensorFlow, Caffe2, PaddlePaddle. 

Yuanquing learned PCA, ?DA? before learning deep learning, laying down the foundations. He also mentioned graphical models and that knowing these gives one very good intuition about how deep learning works. 


-----------

# [Andrej Karpathy](https://www.coursera.org/learn/machine-learning-projects/lecture/Ggkxn/andrej-karpathy-interview)

Started as an undergrad under Hinton at Toronto with Restricted Boltzmann Machines trained on MNIST.

Masters at Uni of Brit Colombia - took a class with Nano? De Freitas where he delved deeper.

Took classes in AI but wasn't satisfying - breadth vs depth first search, alpha/beta pruning etc. 

In contrast, machine learning dealt with neural networks.  The optimisation itself writes code based on the input / output specification.

## ImageNet

Andrej is the human benchmark for the ImageNet image classification competition.  It's the world cup of computer vision. 

Using CIFAR-10, he classified images and got an error rate of about 6%. He predicted the lowest ML error rate would be 10%, but now it's down to about 2 or 3%. 

He then worked on classifying ImageNet into the 1000 categories for about 2 weeks. Not many other people would help him so he didn't get as much data as he wanted. There was some approximate performance number gained though. Andrej is referrerd to as the "reference human" which he finds hilarious.

A third of ImageNet is dogs and dog species.  A third of it's performance comes from dogs!

Andrej was surprised when his performance was surpassed. Sometimes there's a small black dog - which of about 20 categories was just a guess. The ML version gets it easily though. 

A surprising result was bottles of... something. The text on the label showed what it was. Somehow the network learned to read or other methods of identifying what was inside.

## Stanford class

Andrej felt strongly that this technology is transformative and that many people wanted to use it. He felt compelled to hand out hammers (the tech) to students. 

He put his PhD research on hold - it became "120%" of his time. He taught the class twice, 4 months each. Still, he says it was the highlight of his PhD. 

The students loved it as they were discussing papers from only a week ago, or sometimes less, versus results from decades or centuries ago.  

He says it's not like nuclear physics or rocket science - you need to know linear algebra and calculus to undestand everything that happens under the hood. 

## How has his thinking changed?

With RBMs classifying digits it wasn't clear how the tech was going to be used. With conv nets, the thought was that it would never scale to large images. 

Surprised by how general the tech is and how good the results are.  What he and probably everybody else didn't see coming was taking pre-trained networks and training them on arbitrary other tasks. 

Eg a well performing ImageNet NN is an excellent general feature extractor for many tasks. This is called transfer learning.

Many people got into unsupervised learning around 2007, but it's not exactly clear how that is going to work out. 

Has spend the last 1.5 years at OpenAI thinking about the future of AI. He believes the field will split in two:

1. Applied AI - training neural networks, potentially unsupervised learning
2. AGI

Looking back, the way we approached computer vision was wrong: breaking it down into different parts: regognising humans, scenes, objects.

Likewise with AI, we seem to be looking at tasks: Planning, talking, ....  Decomposing by function. 

Andrej thinks that this is an incorrect approach. He believes that creating objectives such that when optimising for the weights that make up the brain it will generate intelligent behaviour.

Suprvised learning - he wrote a short story about a hypothetical future world where based on scaling up supervised learning. 

Unsupervised learning - Algorithmic information theory, eg AIXI, or artificial life. 

## Advice for people entering the field of AI

CS231 praise was that they were not working with a library - the saw the raw code, and implement chunks of it themselves. 

It's important not to abstract away things. Understand the whole stack from the ground up.

Implement everything from scratch was the piece that gave the best bang for buck in terms of understanding. 

He wrote his own library "convnetjs" written in JavaScript to learn about backpropagation. 

Don't work with something like TensorFlow until writing it yourself. 

As of Jan 2018, Andrej was working at Tesla.

