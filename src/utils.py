import matplotlib.pyplot as plt
import numpy as np

class ReduceLROnPlateau:
    """Implements the ReduceLROnPlateau callback.
    Args:
        initial_lr (float): Initial learning rate.
        factor (float): Factor by which learning rate is reduced. Default is 0.1.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default is 10.
        min_lr (float): Minimum learning rate. Default is 1e-6.

    Attributes:
        lr (float): Current learning rate.
        factor (float): Factor for reducing learning rate.
        patience (int): Number of epochs to wait for improvement before reducing learning rate.
        min_lr (float): Minimum learning rate allowed.
        wait (int): Number of epochs with no improvement.
        best_loss (float): Best validation loss observed.
    
    Methods:
        step(validation_loss): Adjusts learning rate based on validation loss.
    """
    def __init__(self, initial_lr, factor=0.1, patience=10, min_lr=1e-6):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')

    def step(self, validation_loss):
        """Adjusts learning rate based on validation loss.

        Args:
            validation_loss (float): Validation loss for the current epoch.
        """
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.lr *= self.factor
                self.lr = max(self.lr, self.min_lr)
                self.wait = 0
                print(f'Reducing learning rate to {self.lr}')
                
                
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
            
            
def plot_losses_from_files(file_paths, error_bar = False, shaded_error = True, show_plot = True, save_path=None):
    """
    Plot and save training and validation losses with average and standard deviation.

    Args:
        file_paths (list of str): List of paths to text files containing the log data.
        error_bar (bool, Optional): Defaults to False. Whether to plot error bars.
        shaded_error (bool, Optional): Defaults to True. Whether to shade the error region.
        show_plot (bool,Optional): Defaults to True. Whether to display the plot.
        save_path (str): Optional path to save the plot.

    Returns:
        None
    """
    data = {}
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        epochs = []
        train_losses = []
        val_losses = []

        for line in lines:
            if 'Epoch' in line:
                parts = line.split('|')
                epoch = int(parts[0].split()[-1])
                train_loss = float(parts[1].split(':')[1])
                val_loss = float(parts[2].split(':')[1])
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

        data[file_path] = {
            'epochs': np.array(epochs),
            'train_losses': np.array(train_losses),
            'val_losses': np.array(val_losses)
        }

    all_epochs = data[file_paths[0]]['epochs']
    all_train_losses = np.zeros((len(file_paths), len(all_epochs)))
    all_val_losses = np.zeros((len(file_paths), len(all_epochs)))

    for i, file_path in enumerate(file_paths):
        log_data = data[file_path]
        all_train_losses[i] = log_data['train_losses']
        all_val_losses[i] = log_data['val_losses']

    avg_train_losses = np.mean(all_train_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)

    avg_val_losses = np.mean(all_val_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot individual training and validation losses from each run
    for i, file_path in enumerate(file_paths):
        # plt.plot(all_epochs, all_train_losses[i], alpha=0.7, label=f'{file_path} - Train Loss', marker='o')
        plt.plot(all_epochs, all_val_losses[i], alpha=0.7, label=f'{file_path} - Val Loss', marker='.')
        plt.legend()
    plt.show()
    if error_bar:
        # Plot average training and validation losses with error bars representing the standard deviation
        plt.errorbar(all_epochs, avg_train_losses, yerr=std_train_losses,
                    label='Average Train Loss', linestyle='-', marker='o', capsize=5)
        plt.errorbar(all_epochs, avg_val_losses, yerr=std_val_losses,
                    label='Average Val Loss', linestyle='-', marker='o', capsize=5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses with Average and Std. Deviation')
        plt.legend()
        
        if show_plot:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
    if shaded_error:
        # Plot average training and validation losses with shaded regions representing the standard deviation
        plt.plot(all_epochs, avg_train_losses, label='Average Train Loss', linestyle='-', marker='o')
        plt.fill_between(all_epochs, avg_train_losses - std_train_losses, avg_train_losses + std_train_losses, alpha=0.3,
                        label='Std. Dev. Train Loss ')
        plt.plot(all_epochs, avg_val_losses, label='Average Validation Loss', linestyle='-', marker='o')
        plt.fill_between(all_epochs, avg_val_losses - std_val_losses, avg_val_losses + std_val_losses, alpha=0.3,
                        label='Std. Dev. Val Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses with Shaded Region for Std. Deviation')
        plt.legend()
        
        if show_plot:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

# # Example usage
# files_to_plot = [r'src\experiment2\results\log1.txt']
# plot_losses_from_files(files_to_plot)
