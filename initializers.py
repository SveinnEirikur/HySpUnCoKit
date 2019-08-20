import numpy as np
import random
from drasl.VCA.VCA import vca
from drasl.SUnSAL.sunsal import sunsal
from pysptools.eea.eea import ATGP as atgp

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def VCA(data, n_end):
    a, i, y = vca(data, n_end)
    m, n = data.shape
    s = np.ones((n, n_end))*1/n_end
    a[a < 1e-7] = 1e-7
    s[s < 1e-7] = 1e-7

    return a, s

def ATGP(data, n_end):
    a, i = atgp(np.transpose(data), n_end)
    m, n = data.shape
    s = np.ones((n, n_end))*1/n_end
    a[a < 1e-7] = 1e-7
    s[s < 1e-7] = 1e-7

    return np.transpose(a), s

def RAND(data, n_end):
    m, n = data.shape
    a = np.random.random((m, n_end))
    s = np.random.random((n, n_end))
    a[a < 1e-7] = 1e-7
    s[s < 1e-7] = 1e-7

    return a, s

def ATGP_srand(data, n_end):
    a, i = atgp(np.transpose(data), n_end)
    m, n = data.shape
    s = np.random.random((n, n_end))
    a[a < 1e-7] = 1e-7
    s[s < 1e-7] = 1e-7

    return np.transpose(a), s

def ATGP_szero(data, n_end):
    a, i = atgp(np.transpose(data), n_end)
    m, n = data.shape
    s = np.zeros((n, n_end))
    a[a < 1e-7] = 1e-7
    s[s < 1e-7] = 1e-7

    return np.transpose(a), s

def ATGP_sunsal(data, n_end):
    a, i = atgp(np.transpose(data), n_end)
    a = np.transpose(a)
    s, res_p, res_d, its = sunsal(a, data, positivity=False, addone=False)
    s = np.transpose(s)
    a[a < 1e-7] = 1e-7
    s[s < 1e-7] = 1e-7

    return a, s
