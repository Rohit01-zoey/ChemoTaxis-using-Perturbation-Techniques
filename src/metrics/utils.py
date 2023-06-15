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
        elif axis==1:
            assert x1.shape[1]!=0
            loss = (1.0/x1.shape[1])*np.sum(np.square(x1 - x2))
        elif axis==2:
            assert x1.shape[2]!=0
            loss = (1.0/x1.shape[2])*np.sum(np.square(x1 - x2))
        else:
            raise NotImplementedError
    else:
        loss = np.sum(np.square(x1 - x2))
    return loss

def mse_loss_seq(x_true, x_pred, batch_norm = True):
    """x_true is the input sequence of the shape (n_samples, n_timestamps, n_features) and x_pred is the predicted sequence of the shape (n_samples, n_timestamps, n_features). 
    x_true = [x_true_1, x_true_2, ..., x_true_n] while x_pred = [x_pred_1, x_pred_2, ..., x_pred_n]. x_pred_1 is the estimate for x_true_2, x_pred_2 is the estimate for x_true_3 and so on. 
    The loss is computed by comparing x_true_2 with x_pred_1, x_true_3 with x_pred_2 and so on.

    Args:
        x_true (_type_): _description_
        x_pred (_type_): _description_
        batch_norm (bool, optional): _description_. Defaults to True.
    
    """
    assert x_true.shape == x_pred.shape
    n_samples, n_timestamps, n_features = x_true.shape

    if batch_norm:
        #take the output of the model y clip the first and last value and the input shd clip the first two values
        loss = (1.0/x_true.shape[0])*np.sum(np.square(x_true[:, 2:, :] - x_pred[:, 1:-1, :]))
    else:
        loss = np.sum(np.square(x_true[:, 2:, :] - x_pred[:, 1:-1, :]))
    return loss

