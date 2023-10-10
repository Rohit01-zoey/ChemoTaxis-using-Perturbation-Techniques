"""
In this file, we will create a sine wave loader class that will load a sinusoidal waves with a given amplitude, frequency and phase.
RNNs are sequence estimators and hence, it is important to load sinusiodal waves with same amplitude and frequency but different phases.
We will also add noise to the sine wave to make the task more challenging.
"""

import numpy as np


class SineWaveLoader:
    """Sine wave loader class loads a single sine wave with a given amplitude, frequency and phase."""
    
    SEED = 42
    
    def __init__(self, n_samples, time_stamps, amplitude, frequency):
        if isinstance(amplitude, (int, float)) and isinstance(frequency, (int, float)):
            # ensure that we input the same amplitudes and frequencies for every phase
            self.n_samples = n_samples # number of samples
            self.time_stamps = time_stamps # number of time stamps
            self.amplitude = amplitude # array of amplitudes
            self.frequency = frequency # array of frequencies
            np.random.seed(self.SEED)
            self.phase = np.random.rand(self.n_samples) * 2 * np.pi # array of phases
        else:
            raise ValueError("Amplitude and frequency should be float values.")

    def load_data(self, attention):
        self.attention = attention # setting the attention of the data and the model
        sine_data = np.zeros((self.n_samples, self.time_stamps+attention,1))
        self.data = {}
        index = 0
        for phase in self.phase:
            timestamps = np.linspace(0, 1, self.time_stamps+attention) # for the last sample extra attention is added
            sine_wave = self.amplitude * np.sin(0.5 * np.pi * self.frequency * timestamps + phase)
            sine_wave = sine_wave.reshape((1, self.time_stamps+attention, 1))
            sine_data[index] = sine_wave
            index += 1 # updating the index to append to data array
        # stacking all the correspoindig sine waves together to get required attention
        if self.attention!=0:
            array_to_stack =[sine_data[:, i:self.time_stamps + i, 0] for i in range(attention+1)]
            sine_data = np.stack(array_to_stack, axis=2)

        random_perm = np.random.permutation(self.n_samples)
        train_data = sine_data[random_perm[:int(0.6 * self.n_samples)], :, :]
        val_data = sine_data[random_perm[int(0.6 * self.n_samples) : int(0.8 * self.n_samples)], :, :]
        test_data = sine_data[random_perm[int(0.8 * self.n_samples) :], :, :]
        sine_data_dict = {'train' : train_data, 'val' : val_data, 'test' : test_data}
        self.data["data"] = sine_data_dict #appending the data to the class
        # return sine_data_dict
    
    def add_noise(self, noise_level):
        """Augments the sine wave with noise.

        Args:
            noise_level (float): The standard deviation of the noise.

        Returns:
            dict: Returns the dictionary of sine waves with noise added/augmented.
        """
        self.augmentation_level = noise_level
        if bool(self.data): # checks that the data is not empty
            self.data['aug'] = {} # dictionary to store the augmented data
            noise = np.random.normal(0, noise_level, (self.n_samples, self.time_stamps, 1))
            noise_dict = {'train' : noise[:int(0.6 * self.n_samples), :, :], 'val' : noise[int(0.6 * self.n_samples) : int(0.8 * self.n_samples), :, :], 'test' : noise[int(0.8 * self.n_samples) :, :, :]}
            for key in self.data["data"].keys():
                self.data["aug"][key] = self.data["data"][key] + noise_dict[key]  # augmented version of the data
        else:
            raise ValueError("Load the data first.")
        # return self.data["aug"]
    
    def get_data(self):
        """Returns the data dictionary.

        Returns:
            dict: Dictionary of the data.
        """
        return self.data






class SineWaveDatasetV2():
    def __init__(self, num_samples= 10, seq_length = 1000, num_periods = 1, frequency_range=2, noise_std=0.01, seed=42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_periods = num_periods
        self.frequency_range = frequency_range
        self.noise_std = noise_std
        self.seed = seed
        self.data = {}
        self.data['data'], self.labels = self._generate_data()

    def _generate_data(self):
        np.random.seed(self.seed)
        data = []
        labels = []

        for _ in range(self.num_samples):
            phase = np.random.uniform(0, 2 * np.pi)
            x = np.linspace(phase, self.num_periods * 2 * np.pi + phase, self.seq_length)
            sine_wave = np.sin(self.frequency_range * x)
            noisy_wave = sine_wave + np.random.normal(0, self.noise_std, self.seq_length)

            data.append([[noisy_wave[i], (self.frequency_range * x[i]) % (2 * np.pi)] for i in range(self.seq_length)])
            labels.append([[noisy_wave[i], (self.frequency_range * x[i]) % (2 * np.pi)] for i in range(self.seq_length)]) # Save the phase as labels

        data = np.array(data)
        labels = np.array(labels)
        
        random_perm = np.random.permutation(self.num_samples)
        train_data = data[random_perm[:int(0.6 * self.num_samples)], :, :]
        val_data = data[random_perm[int(0.6 * self.num_samples) : int(0.8 * self.num_samples)], :, :]
        test_data = data[random_perm[int(0.8 * self.num_samples) :], :, :]
        sine_data_dict = {'train' : train_data, 'val' : val_data, 'test' : test_data}

        return sine_data_dict, labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data['data']['train'][idx], self.labels[idx]


# sine_class = SineWaveLoader(60, 100, amplitude=10.0, frequency=10.0)
# sine_class.load_data(10)
# import matplotlib.pyplot as plt
# plt.plot(sine_class.data['data']['train'][1, :, :])
# plt.show()