# Testing out RNN with one hidden layer at sine wave prediction

## Aim of the experiment

We provide a sinusoidal input to our RNN to train our RNN and next, generate the sine wave using the trained RNN. \
This serves 2 purposes :

1. It verifies that our RNN with `wp` mode of gradient computation works

2. Our RNN model works as a valid sequence generator

## Dataset explained

`src/data/sine_wave.py` generates the sine wave samples for us\
!Add image of sine wave generated

## Experiment details

We choose the following seeds for the weights initialization : 12, 23, ... \

The following learning rate schedule is used : `LearnerRateScheduler(cfg['training']['learning_rate'],0.1, 0.7, 20)`
