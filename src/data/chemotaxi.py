import numpy as np
import pickle

def f(x, y):
        # return 40e-3 + 30e-3 * (np.exp(-(x/20.0 - 3)**2 - (y/20.0 - 3)**2) + np.exp(-(x/20.0 - 4.1)**2 - (y/20.0 - 4.1)**2) - np.exp(-(x/20.0 - 0.5)**2 - (y/20.0 - 0.5)**2))
        # return 300e-3 * y
        # return 400e-3 + 304e-3 * np.exp(-(x/20.0 - 3.5)**2 - (y/20.0 - 2)**2)

        # return 300e-3 * y #1
        # return 30e-3 * x #2
        # return 40.0e-3 + 30.4e-3 * np.exp(-(x/20.0 - 3.5)**2 - (y/20.0 - 2)**2) #3
        # return 40e-3 + 30e-3 * (np.exp(-(x/20.0 - 3)**2 - (y/20.0 - 3)**2) + np.exp(-(x/20.0 - 3)**2 - (y/20.0 - 4.5)**2) + np.exp(-(x/20.0 - 4.1)**2 - (y/20.0 - 4.1)**2) - np.exp(-(x/20.0 - 0.5)**2 - (y/20.0 - 0.5)**2)) #4
        # return ((y<=50)*0.400e-3*y*20 + (y>50)*0.400e-3*(100-y)*20 + np.exp(-(x/20.0 - 3.0)**2 - (y/20.0 - 2.0)**2))*0.05 #6
        return 40e-3 + 30e-3 * (np.exp(-(x/20.0 - 3)**2 - (y/20.0 - 3)**2) + np.exp(-(x/20.0 - 4.1)**2 - (y/20.0 - 4.1)**2) - np.exp(-(x/20.0 - 0.5)**2 - (y/20.0 - 0.5)**2)) #7


class ChemotaxisDataLoader:
    def __init__(self):
        with open('src\datasets\C_Elegans\chemotaxi_data', 'rb') as data:
            dataset_loaded = pickle.load(data)

        with open('src\datasets\C_Elegans\chemotaxi_data_lengths', 'rb') as data:
            self.initial_dataset_length = pickle.load(data)
            
        self.length = min(self.initial_dataset_length)
        self.dataset = {}
        self.dataset['train'] = dataset_loaded[:20, :self.length, :]
        self.dataset['val'] = dataset_loaded[20:, :self.length, :] 
        print("Train Dataset loaded with shape: ", self.dataset['train'].shape)
        print("Validation Dataset loaded with shape: ", self.dataset['val'].shape)
        
        
    def shorten(self, new_length = 200):
        """Shorten the given dataset such that each sample has the same length of new_length. The number
        of training examples increases by old_length//new_length since all these are stacked

        Args:
            new_length (int, optional): The length of the new data. Defaults to 200.
        """
        self.shortened_dataset = {}
        self.shortened_dataset['train'] = np.concatenate([self.dataset['train'][0:1, i*new_length:(i+1)*new_length, :] for i in range(0, self.length//new_length)], axis=0)
        for i in range(1, self.dataset['train'].shape[0]):
            self.shortened_dataset['train'] = np.concatenate([self.shortened_dataset['train'], np.concatenate([self.dataset['train'][i:i+1, j*new_length:(j+1)*new_length, :] for j in range(0, self.length//new_length)], axis=0)], axis = 0)
        
        #for the validation set
        self.shortened_dataset['val'] = np.concatenate([self.dataset['val'][0:1, i*new_length:(i+1)*new_length, :] for i in range(0, self.length//new_length)], axis=0)
        for i in range(1, self.dataset['val'].shape[0]):
            self.shortened_dataset['val'] = np.concatenate([self.shortened_dataset['val'], np.concatenate([self.dataset['train'][i:i+1, j*new_length:(j+1)*new_length, :] for j in range(0, self.length//new_length)], axis=0)], axis = 0)




# data = ChemotaxisDataLoader()
# data.shorten()
# print(data.length)
# print(data.shortened_dataset['train'].shape)