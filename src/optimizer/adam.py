import numpy as np

class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        """ Initialize the Adam optimizer.

        Args:
            lr (float, optional): The learning rate. Defaults to 0.01.
            beta1 (float, optional): The beta1 parameter of the Adam optimizer. Defaults to 0.9.
            beta2 (float, optional): The beta2 parameter of the Adam optimizer. Defaults to 0.999.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads, lr = None):
        """Update the parameters of the model.

        Args:
            params (dict): Dictionary containing the parameters of the model.
            grads (dict): Dictionary containing the gradients of the model.
            lr (float, optional): The learning rate. Defaults to None.
        Returns:
            params (dict): Dictionary containing the updated parameters of the model.
        """
        if lr is not None:
            self.lr = lr
            
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params