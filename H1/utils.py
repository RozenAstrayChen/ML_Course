import numpy as np


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))