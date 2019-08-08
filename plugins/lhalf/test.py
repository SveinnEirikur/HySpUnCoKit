import run_as_plugin
import numpy as np

from pathlib import Path
import pickle
import time
import datetime

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import os

from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
from tqdm import tqdm as tqdm

if __package__ == "lhalf":
    from lhalf.lhalf import lhalf
else:
    from lhalf import lhalf

init_endmem = np.load(Path('../../initials/Samson/endmembers.npy'), allow_pickle=True)
init_abundances = np.load(Path('../../initials/Samson/abundances.npy'), allow_pickle=True)
hsidata = np.load(Path('../../initials/Samson/HSID.npy'), allow_pickle=True)[()]

Y = hsidata.data
ref_endmembers = hsidata.ref_endmembers
init_endmembers = hsidata.init_endmembers
init_abundances = hsidata.init_abundances

resdir = Path('./test', '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now()))
Path.mkdir(resdir, parents=True, exist_ok=True)


def objective(hyperpars):
    results = {}
    A, S, J, SAD, MSE = lhalf(ref_endmembers, init_endmembers,
                         init_abundances, Y, **hyperpars, verbose=True)
    S = S.reshape(hsidata.n_rows, hsidata.n_cols, hsidata.n_endmembers).transpose((1, 0, 2))
    resfile = 'Run_' + str(1) + '.mat'
    outpath = Path(resdir, resfile)
    results.update({'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD, 'MSE': MSE})
    spio.savemat(outpath, results)
    return {'loss': SAD[-1], 'status': STATUS_OK, 'attachments': results}

mu = 0.85
sigma = 1
q = 0.01

space = {
    'max_iter': 500,
    'q': hp.uniform('lhalf_q', 0, 1),
    'delta': hp.uniform('lhalf_delta', 1, 1000),
    'h': [hp.uniform('lhalf_h0', 0, 1000),
          hp.uniform('lhalf_h1', 0, 1000),
          hp.uniform('lhalf_h2', 0, 1000)]
}

trials = Trials()
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)

