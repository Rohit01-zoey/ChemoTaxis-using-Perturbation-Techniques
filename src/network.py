from models import rnn_1hl
from metrics import utils
import logger

import copy
import numpy as np

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
    
def get_gradients(cfg, model, data, loss):
    """Compute the gradients of the model. Returns a ditcionary containing the gradients."""
    if cfg['training']['gradient_computation'] == 'wp':
        #implement weight perturbation
        model_perturbed = copy.deepcopy(model) #deepcopy the model for perturbation
        gradient_dict = copy.deepcopy(get_weights(cfg, model)) #get the dictionary of the weights as placeholder for the loss
        #get the weights of the model
        weights_keys = get_weights(cfg, model_perturbed).keys()
        unperturbed_weights = get_weights(cfg, model)
        for key in weights_keys: #generates the keys of the weight array
            #perturb the weights
            for index, _ in np.ndenumerate(unperturbed_weights[key]): #just for enumerating the indices of the weights array
                weights = get_weights(cfg, model_perturbed) #set the model_perturbed to clean unperturbed weights
                weights[key][index] += cfg['training']['perturbation']
                #set the perturbed weights
                set_weights(cfg, model_perturbed, weights)
                #perform the forward propagation step
                y_perturbed = model_perturbed.forward(data)
                #compute the loss
                loss_perturbed = utils.mse_loss(y_perturbed, data, batch_norm=True, axis= 0)
                #compute the gradient
                gradient_dict[key][index] = (loss_perturbed - loss) / cfg['training']['perturbation']
                #after the gradient is computed, reset the weights to the original weights
                set_weights(cfg, model_perturbed, unperturbed_weights)
        return gradient_dict
    elif cfg['training']['gradient_computation'] == 'bptt':
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
    if cfg['training']['clip']:
        if cfg['model']['name'] == 'rnn_1hl':
            non_clipped_grad =  {'Wxh' : -learning_rate * gradients['Wxh'], 'Whh' : -learning_rate * gradients['Whh'], 'Why' : -learning_rate * gradients['Why'], 'bh' : -learning_rate * gradients['bh'], 'by' : -learning_rate * gradients['by']}
            for key in non_clipped_grad.keys():
                if non_clipped_grad[key].max() > cfg['training']['max_clip_value']:
                    non_clipped_grad[key][non_clipped_grad[key]>cfg['training']['max_clip_value']] = cfg['training']['max_clip_value']
                if non_clipped_grad[key].min() < cfg['training']['min_clip_value']:
                    non_clipped_grad[key][non_clipped_grad[key]<cfg['training']['min_clip_value']] = cfg['training']['min_clip_value']
            return non_clipped_grad

        else:
            raise NotImplementedError
    else:
        if cfg['model']['name'] == 'rnn_1hl':
            return {'Wxh' : -learning_rate * gradients['Wxh'], 'Whh' : -learning_rate * gradients['Whh'], 'Why' : -learning_rate * gradients['Why'], 'bh' : -learning_rate * gradients['bh'], 'by' : -learning_rate * gradients['by']}
        else:
            raise NotImplementedError
    

def train(cfg, model, data, logger):
    """Perform the training step.

    Args:
        cfg (config): config object
        model (model): model object. The model is expected to be initialized!
        data (np.ndarray): data array of shape (n_samples, n_timestamps, n_features)
        logger (logger): logger object to log the training process
    """
    epochs = cfg['training']['epochs']
    learning_rate = cfg['training']['learning_rate']
    # gradient_computation = cfg['training']['gradient_computation']
    for iter in range(epochs):
        #perform the forward propagation step
        output = model.forward(data['train'])
        output_val = model.forward(data['val'])
        #perform the gradient computation step to compute the training loss and validation loss
        train_loss = utils.mse_loss(output, data['train'], batch_norm=True, axis= 0)
        val_loss = utils.mse_loss(output_val, data['val'], batch_norm=True, axis= 0)
        #compute the gradients
        gradients = get_gradients(cfg, model, data['train'], train_loss)
        #get the actual updates from the gradients
        weight_updates = get_weight_updates(cfg, gradients, learning_rate)
        #update the weights
        update_model_weights(cfg, model, weight_updates)
        #log the mse_loss and store it in a file
        logger.log_epoch(iter, train_loss, val_loss)
        #print the loss
        #print("Epoch: {} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(iter, train_loss, val_loss))