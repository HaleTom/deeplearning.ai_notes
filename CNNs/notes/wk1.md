# Convolutional Neural Networks

Computer vision has been advancing rapidly due to Deep Learning.

Examples:

* Self-driving cars
* Facial recognition

DL is even enabling new types of artworks to be created.

What's exciting:

1. Rapid advances allow apps to be built which weren't possible a few years ago
1. Computer vision community is very inventive and creative and ideas can be applied in other areas (Andrew did so with speech recognition)

Even if not working on computer vision, the ideas here will be useful for algorithms and architectures.

## Computer vision problems

* Classification / recognition: Eg, Is this a cat or not?
* Object detection: Where are the objects located?
* Neural Style transfer: repainting a content image in the style of a style image


## Limitation of NNs

Given a 1MP image (3,000,000 values with RGB), and a 1,000 neuron first hidden layer, this will require a `(1_000, 3_000_000)` matrix of weights, or 3 billion parameters.

With that many parameters it's difficult to get enough data to prevent overfitting.

With each parameter being say 4 bytes, that would be 12GB required to represent the weights of only the first hidden layer. The compute requirements will also be infeasible.

To be able to work on larger images, implementing the convolution operation is required.


## Convolutions: edge detection example


