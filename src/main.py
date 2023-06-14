import numpy as np
from models import rnn_1hl

Rnnclass = rnn_1hl.RNN(3, 10, 4)
Rnnclass._intialize()
print(Rnnclass.forward(np.random.randn(2, 3, 3)))