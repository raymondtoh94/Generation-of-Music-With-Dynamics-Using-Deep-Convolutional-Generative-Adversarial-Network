# Generation of Music With Dynamics Using Deep Convolutional Generative Adversarial Network
Source code for https://ieeexplore.ieee.org/abstract/document/9599356

Published in `2021 International Conference on Cyberworlds (CW)`

## Introduction
A Deep learning model will be used to generate expressive music with dynamics. Expressive music requires information such as pitch, time, and velocity, which can be extracted from the MIDI files and encoded in a piano roll format. The piano roll will be used as the symbolic data representation and global step will be used as the temporal scope representation. To generate expressive music with the input of a piano roll, the chosen architecture needs to be able to learn from real samples of expressive music. Hence, a deep convolutional generative adversarial network (DCGAN) is used to fulfil this requirement. It has the ability to learn the data distribution for a given dataset, allowing it to generate expressive music with music dynamics

## Data Pre-Processing
Firstly, a function was written to retrieve all MIDI training data and convert it into a Muspy music object. Next, with the Muspy library, the velocity of all notes was clipped from range 40 to 127 and transposed from its original key into a C major or A minor key. The clipped music object then used Muspy’s built-in function to convert to a piano roll data representation. Also, to improve training time, the piano roll was further constrained with two octaves up and down the starting pitch. Lastly, OpenCV was used to resize the piano-roll into a Nx48x48x1 dimension where N denotes the number of training tracks.

## Data Post-Processing
Before the MIDI generation, the piano roll was processed. Firstly, as generated image produces an intensity (velocity) of -1 to 1, it was rescaled back to the range of 0 to 127. It was further clipped to have a minimum value of 40. Next, the processed piano roll was slotted into an empty tensor of 128x48 (Figure 34). The centre point of the generated piano roll was mapped to 69 of the empty tensors, which was the pitch of A due to the minor scale of the training samples.
