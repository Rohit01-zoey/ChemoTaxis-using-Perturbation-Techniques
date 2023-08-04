import matplotlib.pyplot as plt
import numpy as np

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

    # # Plot individual training and validation losses from each run
    # for i, file_path in enumerate(file_paths):
    #     plt.plot(all_epochs, all_train_losses[i], alpha=0.7, label=f'{file_path} - Train Loss', marker='o')
    #     plt.plot(all_epochs, all_val_losses[i], alpha=0.7, label=f'{file_path} - Val Loss', marker='o')

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
