import numpy as np
import random
from drasl.VCA.VCA import vca
from pysptools.eea.eea import ATGP as atgp

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def VCA(data, n_end):
    a, i, y = vca(data, n_end)
    m, n = data.shape
    s = np.ones((n, n_end))*1/n_end
    return a, s

def ATGP(data, n_end):
    a, i = atgp(data, n_end)
    m, n = data.shape
    s = np.ones((n, n_end))*1/n_end
    return a, s

def RAND(data, n_end):
    m, n = data.shape
    a = np.random.random((m, n_end))
    s = np.random.random((n, n_end))
    return a, s