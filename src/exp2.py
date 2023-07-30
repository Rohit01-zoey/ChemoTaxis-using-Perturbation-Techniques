import numpy as np
import matplotlib.pyplot as plt
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


lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],1, 0.1, 30)
plt.figure()
plt.subplot(121)
plt.plot([i for i in range(1, 1001)],[lr_schedule(i) for i in range(1000)], marker='.', linestyle='-')
plt.title("Learning rate schedule")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid(True)
plt.subplot(122)
plt.semilogx([i for i in range(1, 1001)], [lr_schedule(i) for i in range(1000)], marker='.', linestyle='-')
plt.title("Learning rate schedule (log scale)")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid(True)
plt.show()

import pickle
with open('C:\\Users\\HP\\Downloads\\2020_worm_handoff\\chemotaxi_data', 'rb') as data:
    dataset = pickle.load(data)

with open('C:\\Users\\HP\\Downloads\\2020_worm_handoff\\chemotaxi_data_lenghts', 'rb') as data:
    dataset_length = pickle.load(data)
    
m = min(dataset_length)
dataset_l = {}
dataset_l['train'] = dataset[:1, :m:100, 1:3]
dataset_l['val'] = dataset[2:4, :m:100, 1:3]

cfg = config.get_cfg()
model_name = cfg['model']['name']
input_size = cfg['model']['input_size']
output_size = cfg['model']['output_size']
hidden_size = cfg['model']['hidden_size']
seed = cfg['training']['seed']

log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size, seed=seed)
rnn_model.initialize()

network.train(cfg, rnn_model, dataset_l, logger=log_file)


log_file.close() #free up the logger file

plt.figure()
plt.scatter(dataset_l['train'][0, 2:, 1], dataset_l['train'][0, 2:, 2], marker = '.', color = 'r', label = "True")
plt.scatter(rnn_model.forward(dataset_l['train'])[0, 1:-1, 1], rnn_model.forward(dataset_l['train'])[0, 1:-1, 2], marker = '.', color = 'b', label = "Predicted")
plt.legend()
# plt.savefig()
plt.show()