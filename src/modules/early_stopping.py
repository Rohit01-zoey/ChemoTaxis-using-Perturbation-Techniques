class EarlyStopping:
    """Module for the early stopping technique.
    """
    def __init__(self, patience=5, mode='min'):
        self.patience = patience # the number of epochs to wait before stopping the training process
        self.mode = mode # the mode to compare the loss values. 'min' for loss, 'max' for accuracy
        self.best_score = None # the best score so far
        self.counter = 0 # the number of epochs since the last best score
        self.early_stop = False # flag to stop the training process

        if self.mode == 'min': # initialize the best score
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, val_loss):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        # Save model checkpoint if needed
        
        pass
