from HySpUn import compare_methods
from HSID import HSID
import numpy as np
import scipy.io as spio

datapath='../../Datasets/'
datasets=['Jasper', 'Samson', 'Urban4']
methods=['lhalf', 'matlab_lhalf', 'ACCESSUnmixing']
metrics=['loss', 'SAD', 'endmembers', 'abundances']

Jasper_inits = spio.loadmat('../../Datasets/Jasper_VCA.mat')
init_endmembers = np.array(Jasper_inits['M']).transpose()
init_abundances = np.ones((10000,4))*0.25
jasper = HSID(data_path='../../Datasets/Jasper.mat', dataset_name='Jasper', size=(100,100), n_bands=198,
              n_rows=100, n_cols=100, n_pixels=10000, ref_path='../../Datasets/Jasper_GT.mat', ref_var_names=('M', 'A'),
              init_endmembers=init_endmembers, init_abundances=init_abundances)
hsids = {'Jasper': jasper}

results = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics, hsids=hsids)
