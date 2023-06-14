import numpy as np
import matplotlib.pyplot as plt
from models import rnn_1hl
from metrics import utils
from data import sine_wave
import config
import network
import logger


sine = sine_wave.SineWaveLoader(100, amplitude=1.0, frequency=1.0, phase=0.0)
# plt.plot(sine.load_data()[0, :, 0])
# plt.show()

cfg = config.get_cfg()
model_name = cfg['model']['name']
input_size = cfg['model']['input_size']
output_size = cfg['model']['output_size']
hidden_size = cfg['model']['hidden_size']

log_file = logger.Logger(cfg['log']['log_file'], cfg['log']['experiment_name'])

rnn_model = rnn_1hl.RNN(input_size, hidden_size, output_size)
rnn_model.initialize()

network.train(cfg, rnn_model, sine.load_data(), logger=log_file)

plt.figure()
plt.plot(sine.load_data()['val'][0, :, 0], 'r', label = "True")
plt.plot(rnn_model.forward(sine.load_data()['val'])[0, :, 0], 'b', label = "Predicted")
plt.show()



