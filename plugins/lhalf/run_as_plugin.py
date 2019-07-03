import os
import numpy as np

from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
from tqdm import tqdm as tqdm

if __package__ == "lhalf":
    from lhalf.lhalf import lhalf
else:
    from lhalf import lhalf


#%%
def run_method(hsidata, resdir, num_runs):
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__)))
    configpath = os.path.join(__location__, 'datasets.cfg')
    parser = ConfigParser()
    parser.read(configpath)
    dataset = hsidata.dataset_name
    if not parser.has_section(dataset):
        warn('No settings found for ' + dataset + ', using defaults.')
    q = parser.getfloat(dataset, 'q')
    delta = parser.getfloat(dataset, 'delta')
    h = parser.getfloat(dataset, 'h')
    max_iter = parser.getint(dataset, 'max_iter')
    verbose = parser.getint(dataset, 'verbose')
    Y = hsidata.data
    ref_endmembers = hsidata.ref_endmembers
    init_endmembers = hsidata.init_endmembers
    init_abundances = hsidata.init_abundances
    results = []
    for i in tqdm(range(num_runs), desc="Runs", unit="runs"):
        A, S, J, SAD = lhalf(ref_endmembers, init_endmembers,
                             init_abundances, Y, delta, h, q,
                             max_iter, verbose=verbose)
        resfile = 'Run_' + str(i+1) + '.mat'
        outpath = Path(resdir, resfile)
        results.append({'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD})
        spio.savemat(outpath, results[i])
    with open(Path(resdir, 'datasets.cfg'), 'w') as configfile:
        config.write(configfile)
    return results
