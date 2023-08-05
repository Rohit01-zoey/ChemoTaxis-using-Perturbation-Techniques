"""[ChatGPT : What is Dropout]
Dropout is a form of regularization in deep learning. Regularization techniques are used to prevent overfitting in machine learning models, which occurs when the model performs well on the training data but poorly on unseen data (validation or test data). Overfitting happens when the model captures noise or specific patterns in the training data that do not generalize well to new data.

Dropout is one of the regularization techniques used in neural networks, including RNNs and other deep learning architectures. It was introduced by Srivastava et al. in the paper "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014).

The main idea behind dropout is to randomly "drop out" (set to zero) a fraction of the neurons in the network during training. This means that, during each training iteration, a different set of neurons is active, and the model must learn to make predictions with a subset of neurons missing. This process can be seen as training multiple subnetworks with shared weights.

By using dropout during training, the model becomes less reliant on any specific neuron and cannot rely on complex co-adaptations of neurons. As a result, the model generalizes better to unseen data. At test time, dropout is typically turned off, and the model uses all neurons to make predictions.

Dropout acts as a form of regularization because it introduces noise and uncertainty during training, which helps prevent overfitting. It reduces the risk of memorizing the training data by making the model more robust and less likely to overfit to the specific examples it sees during training.

Other forms of regularization include L1 and L2 regularization, weight decay, early stopping, and data augmentation. These techniques, including dropout, can be used individually or in combination to improve the generalization ability of deep learning models and prevent overfitting."""


import numpy as np

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError('The dropout rate must be between 0 and 1, but got {} as dropout rate.'.format(dropout_rate))

    def forward(self, x, training=True):
        if training and self.dropout_rate != 0:
            self.mask = np.random.rand(*x.shape) < (1 - self.dropout_rate)
            scaled_x = x * self.mask / (1 - self.dropout_rate)
            return scaled_x
        else:
            return x