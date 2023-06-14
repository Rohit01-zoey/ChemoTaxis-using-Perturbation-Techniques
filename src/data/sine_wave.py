import numpy as np


class SineWaveLoader:
    def __init__(self, time_stamps, amplitude=1.0, frequency=1.0, phase=0.0):
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

