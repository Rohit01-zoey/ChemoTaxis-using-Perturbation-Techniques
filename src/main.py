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


sine = sine_wave.SineWaveLoader(n_samples=100, time_stamps=40, amplitude=2.0, frequency=10.0)
sine.load_data() # sine_wave_dict is of the size (train, test, val) x (n_samples, n_timesteps, n_features)
sine.add_noise(noise_level=0.03) # augment the sine wave with noise
sine_wave_dict = sine.get_data() # get the final processed data
plt.plot(sine_wave_dict['data']['train'][1, :, 0], 'r', label = "True sine wave")
plt.plot(sine_wave_dict['aug']['train'][1, :, 0], 'b', label = "Augmented Sine Wave with noise level {}".format(sine.augmentation_level))
plt.legend()
# plt.title("Sine wave with noise")
plt.show()

cfg = config.get_cfg()
model_name = cfg['model']['name']
input_size = cfg['model']['input_size']
output_size = cfg['model']['output_size']
hidden_size = cfg['model']['hidden_size']
seed = cfg['training']['seed']

log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size, seed=seed)
rnn_model.initialize()

network.train(cfg, rnn_model, sine_wave_dict['aug'], logger=log_file)

log_file.close() #free up the logger file

plt.figure()
plt.plot(sine_wave_dict['aug']['test'][0, 1:, 0], 'r', label = "True")
# fsm_model = fsm.FSM(rnn_model)
#generated_sequence = fsm_model(sine_wave_dict['test'][:, 0:1, :], time = 100)
plt.plot(rnn_model.forward(sine_wave_dict['aug']['test'])[0, :, 0], 'b', label = "Predicted")
#plt.plot([generated_sequence[0, i, 0] for i in range(fsm_model.time)], 'b', label = "Predicted")
plt.legend()
plt.show()


fsm_model = fsm.FSM(rnn_model)
generated_sequence = fsm_model(sine_wave_dict['aug']['test'][:, 0:1, :], time = 100)

plt.figure()
plt.plot(sine_wave_dict['aug']['test'][0, :, 0], 'r', label = "True")
plt.plot([generated_sequence[0, i, 0] for i in range(fsm_model.time)], 'b', label = "gen")
plt.show()