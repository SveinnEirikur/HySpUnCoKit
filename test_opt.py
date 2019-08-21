from HySpUn import optimize_methods
from HSID import HSID
import numpy as np
import scipy.io as spio
import random
from initializers import ATGP, ATGP_srand, ATGP_szero, ATGP_sunsal, RAND, VCA

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

datapath = '../../Datasets/'
datasets = ['Jasper', 'Samson', 'Urban4', 'Urban6']
methods = ['lhalf']#, 'ACCESSUnmixing', 'matlab_lhalf']

urban4 = HSID(data_path='../../Datasets/Urban4.mat', dataset_name='Urban4', size=(307,307),
              n_bands=162, n_rows=307, n_cols=307, n_pixels=307*307, ref_var_names=('M', 'A'),
              ref_path='../../Datasets/Urban4_GT.mat')

urban6 = HSID(data_path='../../Datasets/Urban6.mat', dataset_name='Urban6', size=(307,307),
              n_bands=162, n_rows=307, n_cols=307, n_pixels=307*307, ref_var_names=('M', 'A'),
              ref_path='../../Datasets/Urban6_GT.mat')

jasper = HSID(data_path='../../Datasets/Jasper.mat', dataset_name='Jasper',
              size=(100,100), n_bands=198, n_rows=100, n_cols=100, n_pixels=10000, ref_var_names=('M', 'A'),
              ref_path='../../Datasets/Jasper_GT.mat')
hsids = {'Jasper': jasper, 'Urban4': urban4, 'Urban6': urban6}

results = optimize_methods(datasets=datasets, methods=methods, datapath=datapath, hsids=hsids,
                           initializers={'ATGP_sunsal': ATGP_sunsal, 'VCA': VCA})
np.save('./test/08_20_0030.npy', results)
