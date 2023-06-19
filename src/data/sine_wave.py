"""
In this file, we will create a sine wave loader class that will load a sinusoidal waves with a given amplitude, frequency and phase.
RNNs are sequence estimators and hence, it is important to load sinusiodal waves with same amplitude and frequency but different phases.
We will also add noise to the sine wave to make the task more challenging.
"""

import numpy as np


class SineWaveLoader:
    """Sine wave loader class loads a single sine wave with a given amplitude, frequency and phase."""
    
    SEED = 42
    
    def __init__(self, n_samples, time_stamps, amplitude, frequency=1.0):
        if isinstance(amplitude, (int, float)) and isinstance(frequency, (int, float)):
            # ensure that we input the same amplitudes and frequencies for every phase
            self.n_samples = n_samples # number of samples
            self.time_stamps = time_stamps # number of time stamps
            self.amplitude = amplitude # array of amplitudes
            self.frequency = frequency # array of frequencies
            self.phase = np.random.rand(self.n_samples) * 2 * np.pi # array of phases
        else:
            raise ValueError("Amplitude and frequency should be float values.")

    def load_data(self):
        sine_data = np.zeros((self.n_samples, self.time_stamps, 1))
        index = 0
        for phase in self.phase:
            timestamps = np.linspace(0, 1, self.time_stamps)
            sine_wave = self.amplitude * np.sin(2 * np.pi * self.frequency * timestamps + phase)
            sine_wave = sine_wave.reshape((1, self.time_stamps, 1))
            sine_data[index] = sine_wave
            index += 1 # updating the index to append to data array
        random_perm = np.random.permutation(self.n_samples)
        train_data = sine_data[random_perm[:int(0.6 * self.n_samples)], :, :]
        val_data = sine_data[random_perm[int(0.6 * self.n_samples) : int(0.8 * self.n_samples)], :, :]
        test_data = sine_data[random_perm[int(0.8 * self.n_samples) :], :, :]
        sine_data_dict = {'train' : train_data, 'val' : val_data, 'test' : test_data}
        return sine_data_dict
    
    def add_noise(self, noise_level):
        sine_wave = self.load_data()['train']
        new_shape = sine_wave.shape
        new_shape = (self.n_samples, new_shape[1], new_shape[2])
        noise = np.random.normal(0, noise_level, new_shape)
        sine_wave = sine_wave + noise
        sine_wave_dict = {'train' : sine_wave, 'val' : sine_wave, 'test' : sine_wave}
        return sine_wave_dict



# sine_class = SineWaveLoader(60, 100, amplitude=10.0, frequency=10.0)
# data = sine_class.load_data()
# import matplotlib.pyplot as plt
# plt.plot(data['train'][1, :, 0])
# plt.show()