from lhalf import *
from data_load import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


#%%
datadir = '/Volumes/Bengal/Datasets/Jasper'
Y, S_ref, A_ref, n_row, n_col, n_end, n_band = load_data(datadir)
S = np.ones((n_col*n_row,n_end))*0.25
X = np.copy(Y)
q = 0.5
delta = 10
h = 1
max_iter = 5000
verbose = False
Ae = np.load('../../drasl/initial_endmembers.npy')
if verbose:
    A,S,J,SAD,t = lhalf(A_ref,Ae,S,X,delta,h,q,max_iter,verbose)
else:
    A,S = lhalf(A_ref,Ae,S,X,delta,h,q,max_iter,verbose)
spio.savemat('../../output/output.mat',{'A': A, 'S': S})


#%%
# plt.plot(A)
# plt.title(str(max_iter) + " iterations")
# plt.show()
# plt.hist(np.sum(S,axis=1),100)
# plt.title("mean(sum(S)) = " + str(np.mean(np.sum(S,axis=1))))
# plt.show()
# plt.plot(np.diff(J))
# plt.title("max(diff(J)) = " + str(np.max(np.diff(J))))
# plt.show()
# plt.plot(SAD)
# plt.title("min(SAD) = " + str(np.min(SAD)))
# plt.show()


#%%
# S_img = S.reshape((n_row,n_col,n_end))
# S_img = np.transpose(S_img,axes=(1,0,2))
# for abundance in range(S_img.shape[-1]):
#     plt.imshow(S_img[...,abundance])
#     plt.show()
#
# S_ref_img = np.transpose(S_ref,axes=(1,2,0))
# for abundance in range(S_ref_img.shape[-1]):
#     plt.imshow(S_ref_img[...,abundance])
#     plt.show()
