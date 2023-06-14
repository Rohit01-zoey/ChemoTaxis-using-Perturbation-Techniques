from models import rnn_1hl
from metrics import utils

def get_weights(cfg, model):
    """Get the weights of the model.

    Args:
        cfg (config): config object
        model (model): model object

    Raises:
        NotImplementedError: Raises an error if the model name is not supported.

    Returns:
        dict: Returns a dictionary containing the weights of the model.
    """

    if cfg['model']['name'] == 'rnn_1hl':
        return {'Wxh' : model.Wxh, 'Whh' : model.Whh, 'Why' : model.Why, 'bh' : model.bh, 'by' : model.by}
    else:
        raise NotImplementedError
    

def set_weights(cfg, model, weights):
    """Set the weights of the model.

    Args:
        cfg (config): config object
        model (model): model object
        weights (dict): dictionary containing the weights of the model

    Raises:
        NotImplementedError: Raises an error if the model name is not supported.
    """

    if cfg['model']['name'] == 'rnn_1hl':
        model.Wxh = weights['Wxh']
        model.Whh = weights['Whh']
        model.Why = weights['Why']
        model.bh = weights['bh']
        model.by = weights['by']
    else:
        raise NotImplementedError
    
def update_model_weights(cfg, model, weight_updates):
    """Update the weights of the model.

    Args:
        cfg (config): config object
        model (model): model object
        weights_updates (dict): dictionary containing the weights of the model

    Raises:
        NotImplementedError: Raises an error if the model name is not supported.
    """

    if cfg['model']['name'] == 'rnn_1hl':
        model.Wxh += weight_updates['Wxh']
        model.Whh += weight_updates['Whh']
        model.Why += weight_updates['Why']
        model.bh += weight_updates['bh']
        model.by += weight_updates['by']
    else:
        raise NotImplementedError
    
def get_gradients(mode, model, loss):
    """Compute the gradients of the model. Returns a ditcionary containing the gradients."""
    if mode == 'wp':
        #implement weight perturbation
        pass
    elif mode == 'bptt':
        #implement backpropagation through time
        pass
    else:
        raise NotImplementedError
    
def get_weight_updates(cfg, gradients, learning_rate):
    """get the actual weight updates from the gradients.

    Args:
        cfg (config): config object
        gradients (dict): dictionary containing the gradients of the model
        learning_rate (float): the learning rate of the model

    Raises:
        NotImplementedError: Raises an error if the model name is not supported.

    Returns:
        dict: Returns a dictionary containing the actual weight updates of the model.
    """
    if cfg['model']['name'] == 'rnn_1hl':
        return {'Wxh' : -learning_rate * gradients['Wxh'], 'Whh' : -learning_rate * gradients['Whh'], 'Why' : -learning_rate * gradients['Why'], 'bh' : -learning_rate * gradients['bh'], 'by' : -learning_rate * gradients['by']}
    else:
        raise NotImplementedError
    

def train(cfg, model, data):
    """Perform the training step.

    Args:
        cfg (config): config object
        model (model): model object. The model is expected to be initialized!
        data (np.ndarray): data array of shape (n_samples, n_timestamps, n_features)
    """
    epochs = cfg['training']['epochs']
    learning_rate = cfg['training']['learning_rate']
    gradient_computation = cfg['training']['gradient_computation']
    for iter in range(epochs):
        #perform the forward propagation step
        p, h = model.forward(data)
        #perform the gradient computation step
        loss = utils.mse_loss(p, data)
        #compute the gradients
        gradients = get_gradients(gradient_computation, model, loss)
        #get the actual updates from the gradients
        weight_updates = get_weight_updates(cfg, gradients, learning_rate)
        #update the weights
        update_model_weights(cfg, model, weight_updates)
        #log the mse_loss and store it in a file

        pass