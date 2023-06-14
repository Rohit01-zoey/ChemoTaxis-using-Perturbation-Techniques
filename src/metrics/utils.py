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
    """ Sigmoid function that normalizes the output features. !!!!!!!!!CHECK!!!!!!!

    Args:
        x (float): The output features that need to be normalized. Shape: (n_samples, n_features)

    Returns:
        float: Returns the normalized output features. Shape: (n_samples, n_features)
    """
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y_true, y_pred):
    """ Cross entropy loss function.!!!!!!!!!CHECK!!!!!!!

    Args:
        y_true (float): The true labels. Shape: (n_samples, )
        y_pred (float): The predicted labels. Shape: (n_samples, )

    Returns:
        float: Returns the cross entropy loss.
    """
    return -np.sum(y_true * np.log(y_pred))


def mse_loss(x1, x2, batch_norm = True, axis = 0):
    """ Mean squared error loss function.

    Args:
        x1 (float): array like .
        x2 (float): array like. Shapes of x1 and x2 should be same.
        batch_norm (bool, optional): Whether to normalize the loss by the batch size. Defaults to True.
        axis (int, optional): The axis along which the mean of the squared error loss is computed. Defaults to 0.

    Returns:
        float: Returns the mean squared error loss.
    """
    assert x1.shape == x2.shape
    if batch_norm:
        if axis==0:
            assert x1.shape[0]!=0
            loss = (1.0/x1.shape[0])*np.sum(np.square(x1 - x2))
        else:
            assert x1.shape[1]!=0
            loss = (1.0/x1.shape[1])*np.sum(np.square(x1 - x2))
    else:
        loss = np.sum(np.square(x1 - x2))
    return loss