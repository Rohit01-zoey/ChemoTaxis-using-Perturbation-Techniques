import numpy as np

def softmax(x):
    """ Softmax function that normalizes the output features.

    Args:
        x (float): The output features that need to be normalized. Shape: (n_samples, n_features)

    Returns:
        float: Returns the normalized output features. Shape: (n_samples, n_features)
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def sigmoid(x):
    """ Sigmoid function that normalizes the output features.

    Args:
        x (float): The output features that need to be normalized. Shape: (n_samples, n_features)

    Returns:
        float: Returns the normalized output features. Shape: (n_samples, n_features)
    """
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y_true, y_pred):
    """ Cross entropy loss function.

    Args:
        y_true (float): The true labels. Shape: (n_samples, )
        y_pred (float): The predicted labels. Shape: (n_samples, )

    Returns:
        float: Returns the cross entropy loss.
    """
    return -np.sum(y_true * np.log(y_pred))