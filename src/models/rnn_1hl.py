import numpy as np
from metrics import utils

class RNN:
    def __init__(self, input_nodes : int, hidden_nodes : int, output_nodes : int, seed = 42):
        """ Initialize the RNN model."""
        self.input_nodes = input_nodes #the number of input nodes in the RNN
        self.hidden_nodes = hidden_nodes #the number of hidden nodes in each hidden layer
        self.output_nodes = output_nodes #the number of output nodes in the RNN
        self.depth = 3 #the total number of layers in the RNN
        self.seed  = seed #the seed for the random number generator



    def _intialize(self):
        """Intialize the weights and biases of the RNN."""
        np.random.seed(self.seed)
        self.Wxh = np.random.randn(self.input_nodes, self.hidden_nodes) * 0.01 #the weights for the input layer
        self.Whh = np.random.randn(self.hidden_nodes, self.hidden_nodes) * 0.01 #the weights for the hidden layers
        self.Why = np.random.randn(self.hidden_nodes, self.output_nodes) * 0.01 #the weights for the output layer
        self.bh = np.zeros((1, self.hidden_nodes)) #the bias for the hidden layers
        self.by = np.zeros((1, self.output_nodes)) #the bias for the output layer
    
    def forward(self, X : np.ndarray):
        """ Perform the forward propagation step through the RNN.

        Args:
            X (np.ndarray): input data. Shape: (n_samples, n_timesteps, n_features)

        Returns:
            np.ndarray: Returns the probability distribution of the output state. Shape: (n_samples, n_output_nodes) and the hidden state. Shape: (n_samples, n_hidden_nodes)
        """
        #perform the forward propagation step through the RNN
        self.h = np.zeros((X.shape[0], self.hidden_nodes)) #the hidden state
        self.h_prev = np.zeros((X.shape[0], self.hidden_nodes)) #the previous hidden state
        self.y = np.zeros((X.shape[0], self.output_nodes)) #the output state
        self.p = np.zeros((X.shape[0], self.output_nodes)) #the probability distribution of the output state
        self.X = X #the input data
        for t in range(X.shape[1]):
            self.h_prev = self.h
            self.h = np.tanh(np.dot(X[:,t,:], self.Wxh) + np.dot(self.h_prev, self.Whh) + self.bh)
            self.y = np.dot(self.h, self.Why) + self.by
            self.p = utils.softmax(self.y)
        return self.p, self.h

