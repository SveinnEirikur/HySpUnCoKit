import os
import numpy as np
import random
from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
from tqdm import tqdm as tqdm
from HySpUn import mse, improvement_only, save_config

import matlab.engine

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from functools import reduce

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

try:
    @scope.define
    def matlab_double(a):
        try:
            return matlab.double(a)
        except ValueError:
            return matlab.double([a])
except ValueError:
    pass  # it's already defined


#%%
def run_method(hsidata, resdir, num_runs):
    dataset = hsidata.dataset_name

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    mleng = matlab.engine.start_matlab()
    mleng.addpath(__location__)

    configpath = os.path.join(__location__, 'datasets.cfg')
    parser = ConfigParser()
    parser.read(configpath)


    if not parser.has_section(dataset):
        warn('No settings found for ' + dataset + ', using defaults.')
    q = parser.getfloat(dataset, 'q')
    delta = parser.getfloat(dataset, 'delta')
    h = [parser.getfloat(dataset, i) for i in ['h' + str(i) for i in range(hsidata.n_endmembers)]]
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
        S = np.array(output[1]).reshape(hsidata.n_rows, hsidata.n_cols, hsidata.n_endmembers).transpose((1, 0, 2))
        J = np.array(output[2]).tolist()[0]
        SAD = np.array(output[3]).tolist()[0]
        resfile = 'Run_' + str(i+1) + '.mat'
        outpath = Path(resdir, resfile)
        results.append({'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD})
        spio.savemat(outpath, results[i])

    mleng.quit()
    return results


#%%

def opt_method(hsidata, resdir, max_evals):
    dataset_name = hsidata.dataset_name

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    mleng = matlab.engine.start_matlab()
    mleng.addpath(__location__)

    configpath = os.path.join(__location__, 'datasets.cfg')
    parser = ConfigParser()
    parser.read(configpath)
    max_iter = parser.getint(dataset_name, 'max_iter')

    y = hsidata.data
    Y = matlab.double(y.tolist())
    ref_endmembers = matlab.double(hsidata.ref_endmembers.tolist())
    init_endmembers = matlab.double(hsidata.init_endmembers.tolist())
    init_abundances = matlab.double(hsidata.init_abundances.tolist())
    verbose = True

    def objective_func(hyperpars):

        output = mleng.lhalf(ref_endmembers, init_endmembers, init_abundances, Y,
                             hyperpars['q'],
                             hyperpars['delta'],
                             hyperpars['h'],
                             hyperpars['max_iter'],
                             verbose, nargout=5)
        A = np.array(output[0])
        S = np.array(output[1])
        try:
            J = np.array(output[2]).tolist()[0]
        except TypeError:
            J = [output[2]]
        SAD = np.array(output[3]).tolist()[0]

        MSE = mse(y, A, np.transpose(S))

        S = S.reshape(hsidata.n_rows, hsidata.n_cols, hsidata.n_endmembers).transpose((1, 0, 2))

        results = {'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD, 'MSE': MSE}

        return {'loss': SAD[-1], 'status': STATUS_OK, 'attachments': results}

    space = {
        'max_iter': max_iter,
        'q': scope.matlab_double(hp.uniform('lhalf_' + dataset_name + '_q', 0, 1)),
        'delta': scope.matlab_double(hp.uniform('lhalf_' + dataset_name + '_delta', 0, 1000))
    }

    h = scope.matlab_double(
        [hp.uniform('lhalf_' + dataset_name + '_h' + str(i), 0, 1000) for i in range(hsidata.n_endmembers)]
    )

    space['h'] = h

    trials = Trials()

    pars = fmin(lambda x: objective_func(x),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.RandomState(random_seed))

    mleng.quit()

    improvements = reduce(improvement_only, trials.losses(), [])

    save_config(resdir, dataset_name, pars, trials.average_best_error())

    return improvements, pars, trials
