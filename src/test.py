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
from config import cfg
from data.chemotaxi import ChemotaxisDataLoader
from data.data import DataLoader
from utils import plot_losses_from_files, ReduceLROnPlateau, LearnerRateScheduler

# class DirectoryExistsWarning(Warning):
#     pass

# dataset = ChemotaxisDataLoader()

# plt.figure()
# ind = 1
# plt.subplot(121)
# plt.scatter(dataset.dataset['train'][ind, :, 1], dataset.dataset['train'][ind, :, 2], marker = '.', color = 'r', label = "True")
# plt.title("True Location") 
# plt.xlabel("x")
# plt.ylabel("y")
# plt.subplot(122)
# plt.scatter(dataset.dataset['train'][ind, :, 6], dataset.dataset['train'][ind, :, 7], marker = '.', color = 'r', label = "True")
# plt.title("True Updates") 
# plt.xlabel("dx")
# plt.ylabel("dy")
# plt.show()
plot_losses_from_files(['src\log.txt', 'src\log_bp.txt'], error_bar = False, shaded_error = False, show_plot = True, save_path = 'src\\experiment1\\output\\figures\\losses.png')