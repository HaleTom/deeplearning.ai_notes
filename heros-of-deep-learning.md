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

Attention changed Yoshua's view on NNs from somthing that can map a vector to a vector. With attention mechanisms, any type of data structure can be handled. 

In relation to biology, what is like backprop, but could actually be implemented in a brain? Is there a more general princple at play behind backprop? Is there a better way to do credit assignment? 

Unsupervised learning is really important. Supervised learning requires humans to define what the important concepts are and label them. But humans discover new concepts by interactions with the world, eg a 2-year-old understanding phyics.  

He's been combining unsupervised learning with reinforcement learning. In the begining (about 2000), RBMs and autoencoders were focussed on learning good representations (which is still a central problem). But what is a good representation. What's a good objective function which will tell us? [A meta objective function?] 

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

He really only investigates R&D opporunities which have this positive feedback loop as other companies will likely be better at linear development.

## What advice for someone entering AI / DL?

Start with open source frameworks, TensorFlow, Caffe2, PaddlePaddle. 

Yuanquing learned PCA, ?DA? before learning deep learning, laying down the foundations. He also mentioned graphical models and that knowing these gives one very good intuition about how leep learning works. 
