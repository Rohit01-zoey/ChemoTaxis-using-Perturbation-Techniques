import numpy as np

class RMSProp:
    def __init__(self, lr, decay_rate=0.9):
        """ Initialize the RMSProp optimizer.

        Args:
            decay_rate (float, optional): The decay rate of the RMSProp optimizer. Defaults to 0.9.
        """
        self.decay_rate = decay_rate
        self.epsilon = 1e-8
        self.cache = None

    def update(self, params, grads, lr):
        """Update the parameters of the model.

        Args:
            params (dict): Dictionary containing the parameters of the model.
            grads (dict): Dictionary containing the gradients of the model.
            lr (float): The learning rate.
        """
        self.lr = lr
        if self.cache is None:
            self.cache = {}
            for key, val in params.items():
                self.cache[key] = np.zeros_like(val)

        for key in params.keys():
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * (grads[key] ** 2)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return params