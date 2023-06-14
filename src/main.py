import numpy as np
from models import rnn_1hl
from metrics import utils


x1 = np.array([1,2,3])
x2 = np.array([4,5,6])
print(x1.shape, utils.mse_loss(x1, x2,batch_norm=False))