'''
l half unmixing for hyperspectral image data.

Based on Matlab code from Jakob Sigurðsson

Inputs:
Aorg = Array containing reference endmember data (Ground Truth) for comparison,
A = Array containing initialized endmembers,
S = Array containing initialized abundance map,
X = Array containing the hyperspectral data,
delta = Parameter,
h = Parameter,
q = Parameter,
max_iter = Maximum number of iterations,
verbose = Returns more data on each iteration if True,

Outputs:
A = Array containing unmixed endmembers,
S = Array containing unmixed abundance map,
J = List containing loss data
SAD = Array containing SAD data

Python code: Sveinn E. Ármannsson

5. júní 2019, Reykjavík
'''

import numpy as np
from tqdm import tqdm  # _notebook as tqdm
if __package__ == "lhalf":
    from lhalf.calc_SAD import calc_SAD_2
else:
    from calc_SAD import calc_SAD_2


#%%
# Function that computes the S step
def lhalf_S_step(S, ATA, ATX, h, q):
    if q > 0:
        try:
            h = np.diag(h)
        except ValueError:
            h = np.diag([h])
        try:
            S = np.divide(S * ATX, ATA @ S + np.matmul(h, q * np.power(S, q - 1)))
        except ValueError:
            S = np.divide(S * ATX, ATA @ S + h * q * np.power(S, q - 1))
    else:
        S = np.divide(S*ATX, ATA@S)
    S = np.maximum(1e-8, S)
    return S


#%%
# Function that computes the A step
def lhalf_A_step(XST,SST,A):
    A = np.divide(A*XST,A@SST)
    return A


def verbose_plots(i, A, S, J, SAD):
    pass


#%%
# Main function that computes the unmixing
def lhalf(Aorg, A, S, X, delta, h, q=0.5, max_iter=1000, verbose=False):
    J = []
    SAD = np.empty(max_iter)
    S = np.transpose(S)
    M, r = A.shape
    N, P = X.shape
    assert M == N, "Array shape mismatch. A and X should have the same number of lines."
    
    A = np.concatenate((A, delta*np.ones((1, r))))
    X = np.concatenate((X, delta*np.ones((1, P))))
    
    for i in tqdm(range(max_iter)):
        A[0:-1,:] = lhalf_A_step(X[0:-1, :]@np.transpose(S), S@np.transpose(S), A[0:-1, :])
        S = lhalf_S_step(S, np.transpose(A)@A, np.transpose(A)@X, h, q)
        
        if verbose:
            SAD[i] = calc_SAD_2(Aorg, A[0:-1, :])[0]
            
            if i % 10 == 0:
                try:
                    hS = h @ np.power(S, q)
                except ValueError:
                    hS = h * np.power(S, q)
                J.append(0.5*np.sum(np.power(X-A@S, 2))+np.sum(hS))
                if verbose > 1:
                    verbose_plots(i, A, S, J, SAD)
    A = A[0:-1, :]
    S = np.transpose(S)
    if verbose >= 1:
        return A, S, J, SAD
    else:
        return A, S
