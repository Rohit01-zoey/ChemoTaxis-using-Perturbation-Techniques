'''
Implements a RNN model with many hidden layers.
'''

import numpy as np
from metrics import utils
from modules import dropout

class RNNv2:
    def __init__(self, input_nodes: int, hidden_sizes: list[int], output_nodes: int, dropout_rate=0.0, seed=42):
        """
        Initialize the RNN model.

        Args:
            input_nodes (int): The number of input nodes in the RNN.
            hidden_sizes (list): A list of integers representing the number of hidden nodes in each hidden layer.
            output_nodes (int): The number of output nodes in the RNN.
            dropout_rate (float, optional): Dropout rate used for regularization during training. Default is 0.0.
            seed (int, optional): Seed for the random number generator. Default is 42.
        """
        self.name = 'rnn_multilayer'  # name of the model
        self.input_nodes = input_nodes  # the number of input nodes in the RNN
        self.hidden_sizes = hidden_sizes  # a list of hidden layer sizes
        self.output_nodes = output_nodes  # the number of output nodes in the RNN
        self.depth = len(hidden_sizes) + 2  # the total number of layers in the RNN (including input and output layers)
        self.dropout = dropout.Dropout(dropout_rate)  # the dropout layer
        self.seed = seed  # the seed for the random number generator

    def __repr__(self) -> str:
        return f'RNN(input_nodes={self.input_nodes}, hidden_sizes={self.hidden_sizes}, output_nodes={self.output_nodes}, seed={self.seed})'

    def initialize(self):
        """Initialize the weights and biases of the RNN."""
        np.random.seed(self.seed)
        hidden_nodes = [self.input_nodes] + self.hidden_sizes + [self.output_nodes]
        self.weights = []
        self.lateral_weights = []
        self.biases = []
        for i in range(self.depth - 1):
            self.weights.append(np.random.randn(hidden_nodes[i], hidden_nodes[i+1]))
            self.biases.append(np.zeros((1, hidden_nodes[i+1])))
        for i in range(1, self.depth - 1):
            self.lateral_weights.append(np.random.randn(hidden_nodes[i], hidden_nodes[i]))

    def forward(self, X: np.ndarray, training=True):
        """Perform the forward propagation step through the RNN.

        Args:
            X (np.ndarray): input data. Shape: (n_samples, n_timesteps, n_features)

        Returns:
            np.ndarray: Returns the probability distribution of the output state. Shape: (n_samples, n_timesteps, n_output_nodes)
        """
        # perform the forward propagation step through the RNN
        hidden_states = [np.zeros((X.shape[0], size)) for size in self.hidden_sizes]
        self.y = np.zeros((X.shape[0], X.shape[1], self.output_nodes))  # the output state
        self.X = X  # the input data
        for t in range(X.shape[1]):
            h_t = X[:, t, :]
            for i in range(self.depth - 2):
                h_t = np.tanh(np.dot(h_t, self.weights[i]) + np.dot(hidden_states[i], self.lateral_weights[i]) + self.biases[i])
                h_t = self.dropout.forward(h_t, training=training)
                hidden_states[i] = h_t
            self.y[:, t, :] = np.dot(h_t, self.weights[-1]) + self.biases[-1]  # update the output state
        return self.y
