import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from models import rnn_1hl, rnn_xhl
from metrics import utils
from data import sine_wave
import config
import network
import logger
from optimizer import adam, rmsprop
from modules import fsm
from network import LearnerRateScheduler
from config import cfg
from data.chemotaxi import ChemotaxisDataLoader
from data.data import DataLoader
from utils import plot_losses_from_files

class DirectoryExistsWarning(Warning):
    pass

dataset = ChemotaxisDataLoader()
print("Shortening and stacking....")
dataset.shorten(new_length=1000)

dataset_l = dataset.shortened_dataset
dataset_loader = DataLoader(dataset_l['train'], labels = None, batch_size=128, shuffle=True)
print("Shape of the training dataset after shortening: ", dataset_l['train'].shape)
# lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],
#                                    base_learning_rate=0.1,
#                                    final_learning_rate=0.0,
#                                    total_epochs=cfg['training']['epochs'])
# lr_schedule= LearnerRateScheduler(cfg['training']['learning_rate'], 
#                                   base_learning_rate=1, 
#                                   warmup_epochs=5,
#                                   decay_rate = 0.7, 
#                                   decay_steps = 10)
lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'], 
                                   base_learning_rate=0.01)
plt.figure()
plt.subplot(121)
plt.plot([i for i in range(1, cfg['training']['epochs']+1)],[lr_schedule(i) for i in range(cfg['training']['epochs'])], marker='.', linestyle='-')
plt.title("Learning rate schedule")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid(True)
plt.subplot(122)
plt.semilogx([i for i in range(1, cfg['training']['epochs']+1)], [lr_schedule(i) for i in range(cfg['training']['epochs'])], marker='.', linestyle='-')
plt.title("Learning rate schedule (log scale)")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid(True)
plt.show()


cfg = config.get_cfg()
model_name = cfg['model']['name']
input_size = cfg['model']['input_size']
output_size = cfg['model']['output_size']
hidden_size = cfg['model']['hidden_size']
seed = cfg['training']['seed']

root = './src/experiment2/results/{}/'.format(cfg['training']['seed'])
if not os.path.exists(root):
    os.makedirs(root)
else:
    warnings.warn(f"Directory '{root}' already exists. Results will be overwritten.", DirectoryExistsWarning)


log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

if model_name=='rnn_1hl':
    rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size, dropout_rate=0.0, seed=seed)
elif model_name=='rnn_xhl':
    rnn_model = rnn_xhl.RNNv2(input_size, hidden_size, output_size, dropout_rate=0.0, seed=seed)
else:
    raise NotImplementedError

rnn_model.initialize()

network.train_chemotaxis(cfg, rnn_model, dataset_l, lr_schedule, logger=log_file, dataloader = dataset_loader, save_model = True, root=root)


log_file.close() #free up the logger file

plt.show()
plt.figure()
plt.subplot(121)
plt.scatter(dataset_l['train'][0, :, 6], dataset_l['train'][0, :, 7], marker = '.', color = 'r', label = "True")
plt.title("True Updates") 
plt.xlabel("dx")
plt.ylabel("dy")
plt.grid(True)
plt.subplot(122)
plt.scatter(rnn_model.forward(dataset_l['train'][0:1, : , 3:4])[0, :, 0], rnn_model.forward(dataset_l['train'][0:1, : , 3:4])[0, :, 1], marker = '.', color = 'b', label = "Predicted")
plt.title("Predicted Updates")
plt.xlabel("dx")
plt.ylabel("dy")
plt.grid(True)
plt.show()

