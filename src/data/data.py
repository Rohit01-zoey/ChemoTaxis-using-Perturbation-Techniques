import numpy as np

class DataLoader:
    def __init__(self, data, batch_size, shuffle = True):
        self.data = data
        self.batch_size = batch_size
        self.total_samples = data.shape[0]
        self.total_batches = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.current_batch = 0
        if shuffle:
            self.shuffle_indices()

    def shuffle_indices(self):
        self.indices = np.arange(self.total_samples)
        np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.total_batches:
            self.current_batch = 0
            self.shuffle_indices()
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.total_samples)

        batch_indices = self.indices[start_idx:end_idx]
        batch_data = self.data[batch_indices]

        self.current_batch += 1

        return batch_data

    def __len__(self):
        return self.total_batches
