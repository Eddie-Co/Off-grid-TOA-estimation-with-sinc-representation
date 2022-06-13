import numpy as np


def mse(tau, target):
    mse = (np.square(tau - target)).sum()
    return mse


