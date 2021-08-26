import numpy as np

def mean_err(arr):  # shape: (trajs, ...)
    mean = arr.mean(axis=0)
    err = arr.std(axis=0) / np.sqrt(arr.shape[0])  
    return mean, err  


T = 64
V = 24**3

C1, C2 = -0.57462, 1.32789  # Wilson coefficients

M_pi = 0.13975
N_pi = 51.561594
M_K = 0.50365     # \pm 0.0008
N_K = 55.42       # \pm 0.21
Z_V = 0.72672