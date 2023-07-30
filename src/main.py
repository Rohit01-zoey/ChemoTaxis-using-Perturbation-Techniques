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



lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],0.1, 0.7, 20)
plt.figure(figsize=(12, 5))
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
plt.savefig('src\\experiment1\\output\\figures\\learning_rate_schedule.png')
plt.show()

sine = sine_wave.SineWaveLoader(n_samples=100, time_stamps=80, amplitude=4.0, frequency=20.0)
sine.load_data(attention = 0) # sine_wave_dict is of the size (train, test, val) x (n_samples, n_timesteps, n_features)
sine.add_noise(noise_level=0.03) # augment the sine wave with noise
sine_wave_dict = sine.get_data() # get the final processed data
plt.plot(sine_wave_dict['data']['train'][1, :, -1], 'r', label = "True sine wave")
# plt.plot(sine_wave_dict['aug']['train'][1, :, 0], 'b', label = "Augmented Sine Wave with noise level {}".format(sine.augmentation_level))
plt.legend()
# plt.title("Sine wave with noise")
plt.savefig('src\\experiment1\\output\\figures\\input_data.png')
plt.show()

model_name = cfg['model']['name']
input_size = cfg['model']['input_size']
output_size = cfg['model']['output_size']
hidden_size = cfg['model']['hidden_size']
seed = cfg['training']['seed']

log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

lr_schedule = LearnerRateScheduler(cfg['training']['learning_rate'],0.1, 0.7, 20)

rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size, seed=seed)
rnn_model.initialize()

network.train(cfg, rnn_model, sine_wave_dict['data'], lr_schedule, logger=log_file)

log_file.close() #free up the logger file

plt.figure()
plt.plot(sine_wave_dict['data']['test'][0, 2:, -1], 'r', label = "True")
plt.plot(rnn_model.forward(sine_wave_dict['data']['test'])[0, 1:-1, -1], 'b', label = "Predicted")
plt.legend()
plt.title("Input sine wave")
plt.xlabel("Time samples")
plt.ylabel("Amplitude")
plt.savefig('./src/experiment1/output/figures/gen_seq_with_input_full_seq_{}.png'.format(rnn_model.seed))
plt.show()


fsm_model = fsm.FSM(rnn_model)
generated_sequence = fsm_model(sine_wave_dict['data']['test'][:, 0:1, :], time = 100)

plt.figure()
plt.plot(sine_wave_dict['data']['test'][0, :, -1], 'r', label = "True")
plt.plot([generated_sequence[0, i, 0] for i in range(fsm_model.time)], 'b', label = "gen")
plt.legend()
plt.title("Predicted sine wave")
plt.xlabel("Time samples")
plt.ylabel("Amplitude")
plt.savefig('src\\experiment1\\output\\figures\\gen_seq_with_input_first_point_{}.png'.format(rnn_model.seed))
plt.show()
