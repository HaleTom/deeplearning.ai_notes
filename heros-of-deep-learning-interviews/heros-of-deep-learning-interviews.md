
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