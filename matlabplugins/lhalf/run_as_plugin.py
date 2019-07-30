import os
import numpy as np

from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
from tqdm import tqdm as tqdm

import matlab.engine


#%%
def run_method(hsidata, resdir, num_runs):
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__)))
    configpath = os.path.join(__location__, 'datasets.cfg')
    mleng = matlab.engine.start_matlab()
    mleng.addpath(__location__)
    parser = ConfigParser()
    parser.read(configpath)
    dataset = hsidata.dataset_name
    if not parser.has_section(dataset):
        warn('No settings found for ' + dataset + ', using defaults.')
    q = parser.getfloat(dataset, 'q')
    delta = parser.getfloat(dataset, 'delta')
    h = matlab.double([float(i) for i in parser.get(dataset, 'h').split(',')])
    max_iter = parser.getint(dataset, 'max_iter')
    verbose = parser.getboolean(dataset, 'verbose')
    Y = matlab.double(hsidata.data.tolist())
    ref_endmembers = matlab.double(hsidata.ref_endmembers.tolist())
    init_endmembers = matlab.double(hsidata.init_endmembers.tolist())
    init_abundances = matlab.double(hsidata.init_abundances.tolist())
    results = []

    for i in tqdm(range(num_runs), desc="Runs", unit="runs"):
        output = mleng.lhalf(ref_endmembers, init_endmembers, init_abundances, Y, q, delta, h,
                             max_iter, verbose, nargout=5)
        A = np.array(output[0])
        S = np.array(output[1]).reshape(hsidata.n_rows,hsidata.n_cols,hsidata.n_endmembers).transpose((1, 0, 2))
        J = np.array(output[2]).tolist()[0]
        SAD = np.array(output[3]).tolist()[0]
        resfile = 'Run_' + str(i+1) + '.mat'
        outpath = Path(resdir, resfile)
        results.append({'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD})
        spio.savemat(outpath, results[i])

    mleng.quit()
    return results
