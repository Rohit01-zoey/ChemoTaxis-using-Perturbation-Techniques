from typing import Any
import numpy as np

class FSM:
    #implement a finite state machine
    def __init__(self, model, history=1, state = 'generate'):
        self.model = model #intialize the model to be used for the FSM
        self.state = state #initialize the state of the FSM to be 'generate'
        self.history = history

    def __call__(self, X, time = 10):
        """The call method of the FSM class. Generate a sequence of data using the model.

        Args:
            X (np.ndarray): The input data. Shape: (n_samples, n_timesteps = 1, n_features)

        Raises:
            ValueError: If the input data is not 3D of the shape (n_samples, n_timesteps = 1, n_features)
        """
        self.time = time #initialize the time to be 10
        if self.state == 'generate':
            try:
                X.shape[1]==self.history #the number of input data is history number of time step since RNNs assume current sample depends on previous sample
            except:
                raise Exception('The input data must be 3D of the shape ({}, self.history, {}). The shape of the input data is: {}'.format(X.shape[0], X.shape[2], X.shape))
            generated_sequence = np.zeros((X.shape[0], self.time, X.shape[2]))
            generated_sequence[:, :self.history, :] = X[:, :self.history, :] #initialize the first few time-steps of the generated sequence to be the first few time-step of the input data

            #perform the forward propagation step through the RNN
            h = np.zeros((X.shape[0], self.model.hidden_nodes)) #the hidden state
            h_prev = np.zeros((X.shape[0], self.model.hidden_nodes)) #the previous hidden state
            for t in range(1, self.time):
                h_prev = h
                h = np.tanh(np.dot(generated_sequence[:, t-1, :], self.model.Wxh) + np.dot(h_prev, self.model.Whh) + self.model.bh)
                if t>=self.history:
                    generated_sequence[:, t, :] = np.dot(h, self.model.Why) + self.model.by #update the output state
            
            return generated_sequence
        
        elif self.state == 'idle':
            print('The FSM is in the idle state.')

        else:
            raise Exception('The state of the FSM must be "generate". The current state is: {}'.format(self.state))
        

    def set_state(self, state):
        """Set the state of the FSM.

        Args:
            state (str): The state of the FSM. The state can be either 'generate' or 'idle'.
        """
        self.state = state
