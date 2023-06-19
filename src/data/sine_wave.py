"""
In this file, we will create a sine wave loader class that will load a sinusoidal waves with a given amplitude, frequency and phase.
RNNs are sequence estimators and hence, it is important to load sinusiodal waves with same amplitude and frequency but different phases.
We will also add noise to the sine wave to make the task more challenging.
"""

import numpy as np


class SineWaveLoader:
    """Sine wave loader class loads a single sine wave with a given amplitude, frequency and phase."""
    def __init__(self, time_stamps, amplitude, phase, frequency=1.0):
        if isinstance(amplitude, (int, float)) and isinstance(frequency, (int, float)):
            # ensure that we input the same amplitudes and frequencies for every phase
            self.time_stamps = time_stamps # number of time stamps
            self.amplitude = amplitude # array of amplitudes
            self.frequency = frequency # array of frequencies
            self.phase = phase # array of phases
        else:
            raise ValueError("Amplitude and frequency should be float values.")

    def load_data(self):
        sind_data = np.zeros((len(self.phase), self.time_stamps, 1))
        index = 0
        for phase in self.phase:
            timestamps = np.linspace(0, 1, self.time_stamps)
            sine_wave = self.amplitude * np.sin(2 * np.pi * self.frequency * timestamps + phase)
            sine_wave = sine_wave.reshape((1, self.time_stamps, 1))
            sind_data[index] = sine_wave
            index += 1 # updating the index to append to data array
        return sind_data
    
    def add_noise(self, noise_level):
        sine_wave = self.load_data()['train']
        new_shape = sine_wave.shape
        new_shape = (self.n_samples, new_shape[1], new_shape[2])
        noise = np.random.normal(0, noise_level, new_shape)
        sine_wave = sine_wave + noise
        sine_wave_dict = {'train' : sine_wave, 'val' : sine_wave, 'test' : sine_wave}
        return sine_wave_dict



sine_class = SineWaveLoader(100, amplitude= [10, 12], frequency=5, phase=[0.0, 10, 20, 30])
data = sine_class.load_data()
import matplotlib.pyplot as plt
plt.plot(data[1, :, 0])
plt.show()