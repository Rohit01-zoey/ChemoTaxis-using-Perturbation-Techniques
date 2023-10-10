from models import rnn_1hl
from metrics import utils
import logger
from optimizer import adam, rmsprop
from modules import fsm 

from tqdm import tqdm
import time
import copy
import numpy as np
import pickle


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
    elif cfg['model']['name'] == 'rnn_xhl':
        _dict_param = {} #create an empty list
        for i in range(len(model.weights)):
            _dict_param['W{}'.format(i)] = model.weights[i] # append the weights
        for i in range(len(model.lateral_weights)):
            _dict_param['L{}'.format(i)] = model.lateral_weights[i]
        for i in range(len(model.biases)):
            _dict_param['b{}'.format(i)] = model.biases[i]
        return _dict_param
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
    elif cfg['model']['name'] == 'rnn_xhl':
        for key in weights.keys():
            if key[0] == 'W':
                model.weights[int(key[1:])] = weights[key]
            elif key[0] == 'b':
                model.biases[int(key[1:])] = weights[key]
            elif key[0] == 'L':
                model.lateral_weights[int(key[1:])] = weights[key]
            else:
                raise ValueError(f"Unknown key {key}")
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
    elif cfg['model']['name'] == 'rnn_xhl':
        for key in weight_updates.keys():
            if key[0] == 'W':
                model.weights[int(key[1])] += weight_updates[key]
            elif key[0] == 'b':
                model.biases[int(key[1])] += weight_updates[key]
            elif key[0] == 'L':
                model.lateral_weights[int(key[1])] += weight_updates[key]
            else:
                raise ValueError(f"Unknown key {key}")
    else:
        raise NotImplementedError
    
def get_gradients(cfg, model, data, loss, predictions, loss_fn = utils.mse_loss):
    """Compute the gradients of the model. Returns a ditcionary containing the gradients."""
    if cfg['training']['gradient_computation'] == 'wp':
        #implement weight perturbation
        model_perturbed = copy.deepcopy(model) #deepcopy the model for perturbation
        weights_keys = get_weights(cfg, model).keys() #get the keys of the weights
        unperturbed_weights = get_weights(cfg, model) #get the weights of the model
        gradient_dict = copy.deepcopy(get_weights(cfg, model)) #get the dictionary of the weights as placeholder for the loss/gradient
        for key in weights_keys: #generates the keys of the weight array
            #perturb the weights
            for index, _ in np.ndenumerate(unperturbed_weights[key]): #just for enumerating the indices of the weights array
                weights = get_weights(cfg, model_perturbed) #set the model_perturbed to clean unperturbed weights
                weights[key][index] += cfg['training']['perturbation']
                #set the perturbed weights
                set_weights(cfg, model_perturbed, weights)
                #n print(key, index, np.sum(get_weights(cfg, model_perturbed)[key] - unperturbed_weights[key]))
                #perform the forward propagation step
                y_perturbed = model_perturbed.forward(data)
                #compute the loss
                loss_perturbed = utils.mse_loss_seq(y_perturbed, data, batch_norm=True)
                #compute the gradient
                gradient_dict[key][index] = (loss_perturbed - loss) / cfg['training']['perturbation']
                #after the gradient is computed, reset the weights to the original weights
                # set_weights(cfg, model_perturbed, unperturbed_weights)
                del model_perturbed #delete the model_perturbed to free up memory
                model_perturbed = copy.deepcopy(model) #deepcopy the model for perturbation
                # print(key, index, np.sum(get_weights(cfg, model_perturbed)[key] - unperturbed_weights[key]))
        return gradient_dict
    
    elif cfg['training']['gradient_computation'] == 'bptt':
        #implement backpropagation through time
        num_time_steps = data.shape[1]
        # Initialize gradients
        dL_dWhy = np.zeros_like(model.Why)
        dL_dby = np.zeros_like(model.by)
        dL_dWxh = np.zeros_like(model.Wxh)
        dL_dWhh = np.zeros_like(model.Whh)
        dL_dbh = np.zeros_like(model.bh)
        dL_dy = (2/data.shape[0]) * (predictions[:, 1:-1, :] - data[:, 2:, :])
        
        # Initialize gradient for hidden state
        dL_dh_next = np.zeros((dL_dy.shape[0], model.hidden_nodes))
        
        # Backpropagate through time
        for t in reversed(range(1, num_time_steps - 1)):  # Start from 1, not 2.
            dy = dL_dy[:, t-1, :]
            dL_dh = np.dot(dy, model.Why.T) + dL_dh_next
            
            dh_raw = (1 - model.hidden_states[:, t, :]**2) * dL_dh
            
            dL_dWhy += np.dot(model.hidden_states[:, t, :].T, dy)
            dL_dby += dy.sum(axis=0, keepdims=True)
            dL_dWxh += np.dot(data[:, t-1, :].T, dh_raw)  # Use t-1 as input for time step t.
            dL_dWhh += np.dot(model.hidden_states[:, t-1, :].T, dh_raw)
            dL_dbh += dh_raw.sum(axis=0, keepdims=True)
            
            dL_dh_next = np.dot(dh_raw, model.Whh)
        
        # Pack gradients into a dictionary
        gradient_dict = {
            'Why': dL_dWhy,
            'by': dL_dby,
            'Wxh': dL_dWxh,
            'Whh': dL_dWhh,
            'bh': dL_dbh,
        }
    
        return gradient_dict
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
        
        elif cfg['model']['name'] == 'rnn_xhl':
            non_clipped_grad = {}
            for key in gradients.keys():
                non_clipped_grad[key] = -learning_rate * gradients[key]
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
        elif cfg['model']['name'] == 'rnn_xhl':
            non_clipped_grad = {}
            for key in gradients.keys():
                non_clipped_grad[key] = -learning_rate * gradients[key]
        else:
            raise NotImplementedError

    # weight_updates = copy.deepcopy(gradients)
    # for key in gradients.keys():
    #     weight_updates[key] = -learning_rate * gradients[key]
    # return weight_updates
    

def train(cfg, model, data, lr_schedule, logger, dataloader = None, wand = None):
    """Perform the training step.

    Args:
        cfg (config): config object
        model (model): model object. The model is expected to be initialized!
        data (np.ndarray): data array of shape (n_samples, n_timestamps, n_features)
        lr_schedule (callable): learning rate scheduler function. The function should take the current epoch as input and return the learning rate.
        logger (logger): logger object to log the training process
    """
    epochs = cfg['training']['epochs']
    # gradient_computation = cfg['training']['gradient_computation']
    optim = adam.Adam() # initialize the optimizer
    for iter in range(epochs):
        if dataloader is not None:
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {iter + 1}/{epochs}", ncols=140)
            total_train_loss = 0
            for batch in pbar:
                output = model.forward(batch)
                train_loss = utils.mse_loss_seq(output, batch, batch_norm=True)
                total_train_loss += train_loss
                gradients = get_gradients(cfg, model, batch, train_loss) # compute the gradients
                params = optim.update(get_weights(cfg, model), gradients, lr=lr_schedule(iter))
                set_weights(cfg, model, params)
                pbar.set_postfix_str(f"Train loss: {train_loss}")
                time.sleep(0.01) #sleep for 10ms to avoid tqdm progress bar from freezing 
        else:
            output = model.forward(data['train']) # perform the forward propagation step
            train_loss = utils.mse_loss_seq(output, data['train'], batch_norm=True)
            gradients = get_gradients(cfg, model, data['train'], train_loss, output) # compute the gradients
        # if learning_rate == 'auto':
        #     weight_updates = get_weight_updates(cfg, gradients, lr_schedule(iter)) # get the actual updates from the gradients
        # else:
        #     weight_updates = get_weight_updates(cfg, gradients, learning_rate) # get the actual updates from the gradients
            
        # update_model_weights(cfg, model, weight_updates) # update the weights of the model
        params = optim.update(get_weights(cfg, model), gradients, lr=lr_schedule(iter))
        set_weights(cfg, model, params)
        
        output_val = model.forward(data['val'], training=False)
        val_loss = utils.mse_loss_seq(output_val, data['val'], batch_norm=True)
        logger.log_epoch(iter, train_loss if dataloader is None else total_train_loss, val_loss) # log the mse_loss and store it in a file
        if wand is not None:
            # log metrics to wandb
            wand.log({"lr" : lr_schedule(iter), "train loss": train_loss, "test loss": val_loss})
    

def train_chemotaxis(cfg, model, data, lr_schedule, logger, loss_fn = utils.mse_loss,dataloader = None, root = "./", save_model = True):
    """Perform the training step.

    Args:
        cfg (config): config object
        model (model): model object. The model is expected to be initialized!
        data (np.ndarray): data array of shape (n_samples, n_timestamps, n_features)
        lr_schedule (callable): learning rate scheduler function. The function should take the current epoch as input and return the learning rate.
        logger (logger): logger object to log the training process
        dataloader (DataLoader, optional): dataloader object. Defaults to None.
        root (str, optional): root directory to save the model. Defaults to "./".
        save_model (bool, optional): whether to save the model or not. Defaults to True.
    """
    epochs = cfg['training']['epochs']
    
    best_val_loss = np.inf
    
    optim = adam.Adam() # initialize the optimizer
    for iter in range(epochs):
        if dataloader is not None:
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {iter + 1}/{epochs}", ncols=140)
            total_train_loss = 0
            for batch in pbar:
                output = model.forward(batch[:, :, 3:5])
                train_loss = loss_fn(output[:, :-1, :], batch[:, 1:, 7:]*1000, batch_norm=True)
                total_train_loss += loss_fn(output[:, :-1, :], batch[:, 1:, 7:]*1000, batch_norm=False) # adding the true train loss
                gradients = get_gradients(cfg, model, batch, train_loss, loss_fn=loss_fn) # compute the gradients
                params = optim.update(get_weights(cfg, model), gradients, lr=lr_schedule(iter))
                set_weights(cfg, model, params)
                pbar.set_postfix_str(f"Train loss: {train_loss}")
                time.sleep(0.01) #sleep for 10ms to avoid tqdm progress bar from freezing 
            total_train_loss /= data['train'].shape[0] # get the actual averaged train loss
        else:
            output = model.forward(data['train'][:, :, 3:5]) # perform the forward propagation step
            train_loss = loss_fn(output[:, :-1, :], data['train'][:, 1:, 7:]*1000, batch_norm=True)
            gradients = get_gradients(cfg, model, data['train'], train_loss, loss_fn=loss_fn) # compute the gradients

            
        params = optim.update(get_weights(cfg, model), gradients, lr=lr_schedule(iter))
        set_weights(cfg, model, params)
        
        output_val = model.forward(data['val'][:, :, 3:5], training=False)
        val_loss = loss_fn(output_val[:, :-1, :], data['val'][:, 1:, 7:]*1000, batch_norm=True)
        if val_loss <= best_val_loss :
            pickle.dump(model, open(root + "best_model.pkl", "wb")) # save the best model
            
        logger.log_epoch(iter, train_loss if dataloader is None else total_train_loss, val_loss) # log the mse_loss and store it in a file
        
        pickle.dump(model, open(root + "last_model.pkl", "wb")) # save the last model
    
    
