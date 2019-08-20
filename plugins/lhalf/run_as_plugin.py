import os
import numpy as np
import random
from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
from tqdm import tqdm as tqdm

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import reduce

from HySpUn import mse, improvement_only, save_config

from lhalf.lhalf import lhalf

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

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
    h = [parser.getfloat(dataset, i) for i in ['h' + str(i) for i in range(hsidata.n_endmembers)]]
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
        S = S.reshape(hsidata.n_rows,hsidata.n_cols,hsidata.n_endmembers).transpose((1, 0, 2))
        resfile = 'Run_' + str(i+1) + '.mat'
        outpath = Path(resdir, resfile)
        results.append({'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD})
        spio.savemat(outpath, results[i])
    with open(Path(resdir, 'datasets.cfg'), 'w') as configfile:
        parser.write(configfile)
    return results


#%%

def opt_method(hsidata, initializers, resdir, max_evals):
    dataset_name = hsidata.dataset_name

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    configpath = os.path.join(__location__, 'datasets.cfg')
    parser = ConfigParser()
    parser.read(configpath)
    max_iter = parser.getint(dataset_name, 'max_iter')

    def objective_func(hsidata, hyperpars):

        Y = hsidata.data
        ref_endmembers = hsidata.ref_endmembers
        initializer = hyperpars.pop('initializer')
        init_endmembers = initials[initializer][0]
        init_abundances = initials[initializer][1]

        A, S, J, SAD = lhalf(ref_endmembers, init_endmembers,
                             init_abundances, Y, **hyperpars, verbose=True)

        MSE = mse(Y, A, np.transpose(S))
        S = S.reshape(hsidata.n_rows, hsidata.n_cols, hsidata.n_endmembers).transpose((1, 0, 2))
        results = {'endmembers': A, 'abundances': S, 'loss': J, 'SAD': SAD, 'MSE': MSE}
        return {'loss': SAD[-1], 'status': STATUS_OK, 'attachments': results}

    initials = {}
    initial_keys = []
    for key, value in initializers.items():
        initial_keys.append(key)
        initials[key] = (hsidata.initialize(value))

    space = {
        'max_iter': max_iter,
        'q': hp.uniform('lhalf_' + dataset_name + '_q', 0, 1),
        'delta': hp.lognormal('lhalf_' + dataset_name + '_delta', 0, 2),
        'initializer': hp.choice('lhalf_' + dataset_name + '_initializer', initializers)
    }


    h = [hp.lognormal('lhalf_' + dataset_name + '_h' + str(i), 0, 1) for i in range(hsidata.n_endmembers)]

    space['h'] = h

    trials = Trials()

    pars = fmin(lambda x: objective_func(hsidata, x),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.RandomState(random_seed))

    improvements = reduce(improvement_only, trials.losses(), [])

    save_config(resdir, dataset_name, pars, trials.average_best_error())
    print(enumerate(initial_keys))
    return improvements, pars, trials
