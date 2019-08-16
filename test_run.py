from HySpUn import compare_methods
from HSID import HSID
import numpy as np
import random
import scipy.io as spio
from initializers import VCA

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

datapath = '../../Datasets/'
datasets = ['Jasper']#, 'Samson', 'Urban4']
methods = ['lhalf']#, 'ACCESSUnmixing', 'matlab_lhalf']
metrics = ['loss', 'SAD', 'endmembers', 'abundances']

Jasper_inits = spio.loadmat('../../Datasets/Jasper_VCA.mat')
init_endmembers = np.array(Jasper_inits['M']).transpose()
init_abundances = np.ones((10000,4))*0.25
jasper_bands = [i for i in range(224)]
skipped_bands = [i for j in (range(0,3), range(107,112), range(153,166), range(219,224)) for i in j]
jasper_bands = [np.nan if i in skipped_bands else i for i in jasper_bands]
jasper = HSID(data_path='../../Datasets/Jasper.mat', dataset_name='Jasper', size=(100,100), n_bands=198,
              n_rows=100, n_cols=100, n_pixels=10000, ref_path='../../Datasets/Jasper_GT.mat', ref_var_names=('M', 'A'),
              freq_list=np.floor(np.linspace(380,2500,224)), bands_to_use=jasper_bands)
hsids = {'Jasper': jasper}

results = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics,
                          hsids=hsids, initializer=VCA)
