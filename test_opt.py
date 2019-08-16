from HySpUn import optimize_methods
from HSID import HSID
import numpy as np
import scipy.io as spio
import random
from initializers import VCA

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

datapath = '../../Datasets/'
datasets = ['Samson', 'Jasper', 'Urban4', 'Urban6']
methods = ['lhalf']#, 'ACCESSUnmixing', 'matlab_lhalf']

Jasper_inits = spio.loadmat('../../Datasets/Jasper_VCA.mat')
init_endmembers = np.array(Jasper_inits['M']).transpose()
init_abundances = np.ones((10000,4))*0.25
jasper = HSID(data_path='../../Datasets/Jasper.mat', dataset_name='Jasper', size=(100,100), n_bands=198,
              n_rows=100, n_cols=100, n_pixels=10000, ref_path='../../Datasets/Jasper_GT.mat', ref_var_names=('M', 'A'),
              init_endmembers=init_endmembers, init_abundances=init_abundances)
hsids = {'Jasper': jasper}

results = optimize_methods(datasets=datasets, methods=methods, datapath=datapath, hsids=hsids, initializer=VCA)
np.save('./test/08_16_1400.npy', results)
