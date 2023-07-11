# C. Elegans Movement
We wish to simulate the movement of the C. Elegans in a concetration gradient. We employ a Recurrent Neural Network (RNN) to simulate this behaviour.

# Motivation to use RNNs
As seen in Experiment 1, we use a RNN to estimate a sine wave. This motivates us to use a RNN as a sequence generator for C. Elegans.

# Input data
[Code]() is used to generate the sequence for different initial points  given a concentration curve. Thus, our data will be fo the shape (n_samples, n_timestamps, n_features). Here the n_features will contain [$\theta$, x, y, v, $C_T$]