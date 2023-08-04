import numpy as np
import matplotlib.pyplot as plt
import wandb


from models import rnn_1hl
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

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Chemotaxis",
    
#     # track hyperparameters and run metadata
#     config={
#     "architecture": "RNN",
#     "dataset": "Chemotaxis",
#     "epochs": 10,
#     "hidden layers" : 20
#     }
# )

dataset = ChemotaxisDataLoader()
print("Shortening and stacking....")
dataset.shorten(200)

dataset_l = dataset.shortened_dataset
dataset_loader = DataLoader(dataset_l['train'], batch_size=128, shuffle=True)
print("Shape of the training dataset after shortening: ", dataset_l['train'].shape)
#lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],2, 0.1, 20)
lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],
                                   base_learning_rate=0.01,
                                   final_learning_rate=0.0,
                                   total_epochs=cfg['training']['epochs'])
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
# lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'], 1, 0.7, 20)
lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],
                                   base_learning_rate=1,
                                   final_learning_rate=0.0,
                                   total_epochs=cfg['training']['epochs'])
log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size, seed=seed)
rnn_model.initialize()

network.train(cfg, rnn_model, dataset_l, lr_schedule, logger=log_file, dataloader = dataset_loader, wand=None)
# [optional] finish the wandb run, necessary in notebooks
# wandb.finish()


log_file.close() #free up the logger file

plt.figure()
plt.scatter(dataset_l['train'][0, 2:, 1], dataset_l['train'][0, 2:, 2], marker = '.', color = 'r', label = "True")
plt.scatter(rnn_model.forward(dataset_l['train'])[0, 1:-1, 1], rnn_model.forward(dataset_l['train'])[0, 1:-1, 2], marker = '.', color = 'b', label = "Predicted")
plt.legend()
# plt.savefig()
plt.show()

