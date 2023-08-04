from models import rnn_1hl
from metrics import utils
import logger
from tqdm import tqdm
import time

from optimizer import adam, rmsprop
from modules import fsm 

import copy
import numpy as np

class LearnerRateScheduler:
    """Learning rate scheduler class. This class implements the learning rate scheduler."""
    def __init__(self, type, base_learning_rate, warmup_epochs=10, **kwargs):
        """_summary_

        Args:
            type (str): The type of the learning rate scheduler. Can be one of 'constant', 'linear', 'exponential' or 'step'.
            base_learning_rate (float): The learning rate to start with after warmup_epochs.
            warmup_epochs (int, optional): The number of epochs for warm-up. Linear in nature. Goes from lr_init(defaults to 0) to base learning rate . Defaults to 10.
            **kwargs: Additional arguments for the learning rate scheduler. The arguments depend on the type of scheduler used.
            
            
        Raises:
            TypeError: _description_
            TypeError: _description_
            TypeError: _description_
            TypeError: _description_
            TypeError: _description_
        """
        self.type = type
        self.base_learning_rate = base_learning_rate
        allowed_parameters = ['final_learning_rate', 'decay_rate', 'decay_steps', 'total_epochs', 'lr_init']
        # Check if any unknown keys are present in kwargs
        unknown_parameters = set(kwargs.keys()) - set(allowed_parameters)
        if unknown_parameters:
            raise TypeError(f"Unknown parameter(s) provided: {', '.join(unknown_parameters)}")
        self.final_learning_rate = kwargs['final_learning_rate'] if 'final_learning_rate' in kwargs else None
        self.decay_rate = kwargs['decay_rate'] if 'decay_rate' in kwargs else None
        self.decay_steps = kwargs['decay_steps'] if 'decay_steps' in kwargs else None
        self.warmup_epochs = warmup_epochs
        self.total_epochs = kwargs['total_epochs'] if 'total_epochs' in kwargs else None
        self.lr_init = kwargs['lr_init'] if 'lr_init' in kwargs else 0.0
        
        if self.type == 'linear':
            if 'final_learning_rate' not in kwargs.keys():
                raise TypeError(f"final_learning_rate must be provided for linear decay")
            if 'total_epochs' not in kwargs.keys():
                raise TypeError(f"total_epochs must be provided for linear decay")
        if self.type == 'step':
            if 'decay_rate' not in kwargs.keys():
                raise TypeError(f"decay_rate must be provided for step decay")
            if 'decay_steps' not in kwargs.keys():
                raise TypeError(f"decay_steps must be provided for step decay")
    
    def __call__(self, step):
        if step < self.warmup_epochs:
            #linear increase to base_learning_rate
            return self.lr_init + (self.base_learning_rate-self.lr_init) * (step / self.warmup_epochs)
        else:
            if self.type == 'constant':
                return self.base_learning_rate
            elif self.type == 'linear':
                return self.base_learning_rate - (self.base_learning_rate - self.final_learning_rate) * (step - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            elif self.type == 'exponential':
                pass
            elif self.type == 'step':
                return self.base_learning_rate * (self.decay_rate ** (int(step / self.decay_steps)))
            else:
                raise NotImplementedError
            

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
            gradients = get_gradients(cfg, model, data['train'], train_loss) # compute the gradients
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
    
