from HySpUn import compare_methods
from HSID import HSID
import numpy as np
import random
import scipy.io as spio
from initializers import ATGP_sunsal

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

datapath = '../../Datasets/'
datasets = ['Jasper', 'Samson', 'Urban4', 'Urban6']
methods = ['matlab_lhalf', 'lhalf', 'ACCESSUnmixing']
metrics = ['loss', 'SAD', 'endmembers', 'abundances']

jasper_bands = [i for i in range(224)]
skipped_bands = [i for j in (range(0,3), range(107,112), range(153,166), range(219,224)) for i in j]
jasper_bands = [np.nan if i in skipped_bands else i for i in jasper_bands]

urban_bands = [i for i in range(210)]
skipped_bands = [i for j in (range(0,4), range(75,76), range(86,87), range(100,111), range(135,153), range(197,210)) for i in j]
urban_bands = [np.nan if i in skipped_bands else i for i in urban_bands]

jasper = HSID(data_path='../../Datasets/Jasper.mat', dataset_name='Jasper',
              size=(100,100), n_bands=198, n_rows=100, n_cols=100, n_pixels=10000, ref_var_names=('M', 'A'),
              ref_path='../../Datasets/Jasper_GT.mat',
              freq_list=np.floor(np.linspace(380,2500,224)), bands_to_use=jasper_bands)

urban4 = HSID(data_path='../../Datasets/Urban4.mat', dataset_name='Urban4', size=(307,307),
              n_bands=162, n_rows=307, n_cols=307, n_pixels=307*307, ref_var_names=('M', 'A'),
              ref_path='../../Datasets/Urban4_GT.mat',
              freq_list=np.floor(np.linspace(400,2500,210)), bands_to_use=urban_bands)

urban6 = HSID(data_path='../../Datasets/Urban6.mat', dataset_name='Urban6', size=(307,307),
              n_bands=162, n_rows=307, n_cols=307, n_pixels=307*307, ref_var_names=('M', 'A'),
              ref_path='../../Datasets/Urban6_GT.mat',
              freq_list=np.floor(np.linspace(400,2500,210)), bands_to_use=urban_bands)

hsids = {'Jasper': jasper, 'Urban4': urban4, 'Urban6': urban6}

results = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics,
                          hsids=hsids, initializer=ATGP_sunsal)
