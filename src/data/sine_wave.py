import numpy as np


class SineWaveLoader:
    """Sine wave loader class loads a single sine wave with a given amplitude, frequency and phase."""
    def __init__(self, n_samples,time_stamps, amplitude=1.0, frequency=1.0, phase=0.0):
        self.n_samples = n_samples
        self.time_stamps = time_stamps
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def load_data(self):
        timestamps = np.linspace(0, 1, self.time_stamps)
        sine_wave = self.amplitude * np.sin(2 * np.pi * self.frequency * timestamps + self.phase)
        sine_wave = sine_wave.reshape((1, self.time_stamps, 1))
        sine_wave_dict = {'train' : sine_wave, 'val' : sine_wave, 'test' : sine_wave}
        return sine_wave_dict
    
    def add_noise(self, noise_level):
        sine_wave = self.load_data()['train']
        new_shape = sine_wave.shape
        new_shape = (self.n_samples, new_shape[1], new_shape[2])
        noise = np.random.normal(0, noise_level, new_shape)
        sine_wave = sine_wave + noise
        sine_wave_dict = {'train' : sine_wave, 'val' : sine_wave, 'test' : sine_wave}
        return sine_wave_dict


