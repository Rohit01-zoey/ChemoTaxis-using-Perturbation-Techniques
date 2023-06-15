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


sine = sine_wave.SineWaveLoader(60, 100, amplitude=10.0, frequency=1.0, phase=0.0)
sine_wave_dict = sine.load_data()
sine_wave_dict = sine.add_noise(1)
# plt.plot(sine_wave_dict['train'][1, :, 0])
# plt.show()

cfg = config.get_cfg()
model_name = cfg['model']['name']
input_size = cfg['model']['input_size']
output_size = cfg['model']['output_size']
hidden_size = cfg['model']['hidden_size']
seed = cfg['training']['seed']

log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size, seed=seed)
rnn_model.initialize()

network.train(cfg, rnn_model, sine.load_data(), logger=log_file)

log_file.close() #free up the logger file

plt.figure()
plt.plot(sine.load_data()['val'][0, :, 0], 'r', label = "True")
plt.plot(rnn_model.forward(sine.load_data()['val'])[0, :, 0], 'b', label = "Predicted")
plt.legend()
plt.show()


fsm_model = fsm.FSM(rnn_model)
fsm_model(np.random.rand(1, 1, 1), time = 100)

plt.figure()
plt.plot([fsm_model.generated_sequence[0, i, 0] for i in range(fsm_model.time)], 'r', label = "gen")
plt.show()