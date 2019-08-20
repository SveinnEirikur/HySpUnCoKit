import sys
import datetime
import numpy as np
from configparser import ConfigParser
from tqdm import tqdm
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from HSID import HSID
from itertools import product
from matplotlib import pyplot as plt
import seaborn as sns
import random

from tools.calc_SAD import calc_SAD_2

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def compare_methods(datasets: list, methods: list, datapath: str = None, hsids: dict = None,
                    initializer: staticmethod = None, metrics_to_plot: list = None, plot: bool = True) -> dict:
    """A method for comparing hyperspectral unmixing methods

    Extra parameters for each method, such as the path to the method, are accessed from the local methods.cfg file.
    Further parameters should be set on a per dataset basis in its run_as_plugin.py and datasets.cfg.

    Either datapath to load .mat files into HSID objects or dict with HSID objects must be provided.

    :param datasets: list of strings of dataset names without .mat extensions, also assumed to be filenames
                    accessed at datapath/dataset.mat if dataset key is not present in hsids dict.
    :param methods: list of strings of method names, names and paths should be added to methods.cfg as well.
    :param datapath: string that contains the directory path where .mat data should be accessed.
    :param hsids: dict of HSID objects containing Hyper Spectral Image Data to be processed with dataset names as keys.

    Optional:

    :param initializer: a function that takes as two arguments: the data and required number of endmembers, and returns
                        initial endmembers and abundances in that order
    :param plot: boolean which if true the method saves png plots based on method results and metrics.
    :param metrics_to_plot: list of strings of metric names to plot.

    :return: dict object containing the results of all methods, datasets and runs.
    """
    if hsids is None:
        hsids = {}

    if metrics_to_plot is None:
        metrics_to_plot = ['endmembers', 'loss', 'SAD']

    initials_dir = Path('./initials')

    config = ConfigParser()
    config.read('methods.cfg')

    for dataset in datasets:
        if dataset not in hsids:
            hsids[dataset] = HSID(datapath + dataset + '.mat', dataset_name=dataset)
        if initializer is not None:
            hsids[dataset].initialize(initializer)

    n_runs = config.getint('DEFAULT', 'n_runs', fallback=1)

    print('Running methods: ')
    print(*methods, sep=", ")
    print('On the following datasets: ')
    print(*datasets, sep=", ")
    print('Number of runs per method:', n_runs)

    timestamp = '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())

    results = {key: {k: [] for k in methods} for key in datasets}

    for dataset in tqdm(datasets, desc='Datasets', unit='sets'):
        for method in tqdm(methods, desc='Methods', unit='method'):
            package_path = Path(config[method]['path'])
            results_path = Path(config[method]['output'])
            module_path = Path(package_path, 'run_as_plugin.py')
            spec = spec_from_file_location(method, module_path, submodule_search_locations=[])
            plugin = module_from_spec(spec)
            sys.modules[spec.name] = plugin
            spec.loader.exec_module(plugin)

            respath = Path(results_path, timestamp, dataset, method)
            Path.mkdir(respath, parents=True, exist_ok=True)

            output = plugin.run_method(hsids[dataset], respath, n_runs)
            results[dataset][method] = output

    for dataset in datasets:
        if hsids[dataset].ref_endmembers is not None:
            results[dataset]['ref_endmembers'] = hsids[dataset].ref_endmembers
        if hsids[dataset].ref_abundances is not None:
            results[dataset]['ref_abundances'] = hsids[dataset].ref_abundances

    with open(Path(config['DEFAULT']['output'], timestamp, 'methods.cfg'), 'w') as configfile:
        config.write(configfile)

    np.save(Path(config['DEFAULT']['output'], timestamp, 'Results.npy'), results, allow_pickle=True)

    if plot:
        plot_results(results, hsids, metrics_to_plot, datasets, methods,
                     n_runs, Path(config['DEFAULT']['output'], timestamp))

    return results


def optimize_methods(datasets: list, methods: list, datapath: str = None, hsids: dict = None,
                     initializers: dict = None,) -> dict:
    """A method for optimizing hyperparameters of hyperspectral unmixing methods

    Extra parameters for each method, such as the path to the method, are accessed from the local methods.cfg file.
    Further parameters should be set on a per dataset basis in its run_as_plugin.py and datasets.cfg.

    Either datapath to load .mat files into HSID objects or dict with HSID objects must be provided.

    :param datasets: list of strings of dataset names without .mat extensions, also assumed to be filenames
                    accessed at datapath/dataset.mat if dataset key is not present in hsids dict.
    :param methods: list of strings of method names, names and paths should be added to methods.cfg as well.

    Optional:
    :param datapath: string that contains the directory path where .mat data should be accessed.
    :param hsids: dict of HSID objects containing Hyper Spectral Image Data to be processed with dataset names as keys.
    :param initializer: a function that takes as two arguments: the data and required number of endmembers, and returns
                        initial endmembers and abundances in that order

    :return: dict of Hyperopt trials objects containing the results of all methods and datasets.
    """
    if hsids is None:
        hsids = {}

    initials_dir = Path('./initials')

    config = ConfigParser()
    config.read('methods.cfg')

    for dataset in datasets:
        if dataset not in hsids:
            hsids[dataset] = HSID(datapath + dataset + '.mat', dataset_name=dataset)
    max_evals = config.getint('DEFAULT', 'max_evals', fallback=100)

    print('Running methods: ')
    print(*methods, sep=", ")
    print('On the following datasets: ')
    print(*datasets, sep=", ")
    print('Maximum evaluations per method:', max_evals)

    timestamp = '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())

    results = {key: {k: [] for k in methods} for key in datasets}

    for method in tqdm(methods, desc='Methods', unit='method'):
        for dataset in tqdm(datasets, desc='Datasets', unit='sets'):
            package_path = Path(config[method]['path'])
            results_path = Path(config[method]['output'])
            module_path = Path(package_path, 'run_as_plugin.py')
            spec = spec_from_file_location(method, module_path, submodule_search_locations=[])
            plugin = module_from_spec(spec)
            sys.modules[spec.name] = plugin
            spec.loader.exec_module(plugin)

            respath = Path(results_path, timestamp, dataset, method)
            Path.mkdir(respath, parents=True, exist_ok=True)

            loss, pars, trials = plugin.opt_method(hsids[dataset], initializers, respath, max_evals)
            results[dataset][method] = {'best loss': min(loss), 'best parameters': pars,
                                        'loss progression': loss, 'trials': trials}

    for dataset, method in product(datasets, methods):
        print('{} {} Best loss: {}\nParameters: {}'.format(dataset, method,
                                                           results[dataset][method]['best loss'],
                                                           results[dataset][method]['best parameters']))

    return results


def plot_results(results, hsids, metrics, datasets, methods, n_runs, respath):
    """

    :param results:
    :param metrics:
    :param datasets:
    :param methods:
    :param n_runs:
    :param respath:
    :return:
    """
    sns.set_style('darkgrid')
    sns.set_context("talk")

    m_sad = {dataset: [[] for i in range(len(methods))] for dataset in datasets}
    m_idx_hat = {dataset: [[]for i in range(len(methods))] for dataset in datasets}

    if 'loss' in metrics:

        for dataset in datasets:
            fig, axes = plt.subplots(nrows=n_runs,
                                     ncols=len(methods),
                                     figsize=(5 * len(methods),
                                              5 * n_runs))

            for ax, (run, method) in zip(axes.flatten(), product(range(n_runs), methods)):
                ax.plot(results[dataset][method][run]['loss'])
                ax.set(title="{},\n{} run: {}".format(method, dataset, run+1))

            fig.subplots_adjust(hspace=0.5)
            plt.savefig(Path(respath, dataset + '_loss.png'), dpi=200, bbox_inches='tight', format='png')
            plt.clf()

    if 'SAD' in metrics:

        for dataset in datasets:
            fig, axes = plt.subplots(nrows=n_runs,
                                     ncols=len(methods),
                                     figsize=(5 * len(methods),
                                              5 * n_runs))

            for ax, (run, method) in zip(axes.flatten(), product(range(n_runs), methods)):
                ax.plot(results[dataset][method][run]['SAD'])
                ax.set(title="{},\n{} run: {}".format(method, dataset, run+1))

            fig.subplots_adjust(hspace=0.5)
            plt.savefig(Path(respath, dataset + '_SAD.png'), dpi=200, bbox_inches='tight', format='png')
            plt.clf()

    if 'endmembers' in metrics:

        for dataset in datasets:
            n_bands, n_endmembers = results[dataset]['ref_endmembers'].shape
            fig, axes = plt.subplots(nrows=len(methods) + 1,
                                     ncols=n_endmembers,
                                     figsize=(7 * n_endmembers,
                                              6 * len(methods) + 6),
                                     sharey='row', sharex='col')

            # Plot reference endmembers
            if 'ref_endmembers' in results[dataset]:
                for n in range(n_endmembers):
                    ref_endmember = results[dataset]['ref_endmembers'][:, n]
                    if hsids[dataset].bands_to_use is not None:
                        for i in range(len(hsids[dataset].bands_to_use)):
                            if hsids[dataset].bands_to_use[i] is np.nan:
                                ref_endmember = np.insert(ref_endmember,i,np.nan)
                    if hsids[dataset].freq_list is not None:
                        freq_list = hsids[dataset].freq_list
                    else:
                        freq_list = [i for i in range(ref_endmember.shape[0])]
                    axes[0][n].plot(freq_list, ref_endmember)
                    axes[0][n].set(title="{},\n{} endmember: {}".format(dataset, 'reference', n + 1))
                    axes[0][n].fill_between(freq_list, np.nanmin(ref_endmember), np.nanmax(ref_endmember),
                                            where=np.isnan(ref_endmember), color='grey', alpha=0.5)
            # Plot every runs endmembers for each methods in apropriate column
            linecolor = sns.color_palette()[0]
            for (midx, eidx) in product(range(len(methods)), range(n_endmembers)):
                for run in range(n_runs):
                    sad_m, idx_org_m, idx_hat_m, sad_k_m, s0 = \
                        calc_SAD_2(results[dataset]['ref_endmembers'],
                                   results[dataset][methods[midx]][run]['endmembers'])
                    m_sad[dataset][midx].append(sad_m)
                    m_idx_hat[dataset][midx].append(idx_hat_m)
                    endmember = results[dataset][methods[midx]][run]['endmembers'][:, idx_hat_m[eidx]]
                    if hsids[dataset].bands_to_use is not None:
                        for i in range(len(hsids[dataset].bands_to_use)):
                            if hsids[dataset].bands_to_use[i] is np.nan:
                                endmember = np.insert(endmember,i,np.nan)
                    if hsids[dataset].freq_list is not None:
                        freq_list = hsids[dataset].freq_list
                    else:
                        freq_list = [i for i in range(len(endmember))]
                    axes[midx+1][eidx].plot(freq_list, endmember,
                                            color=linecolor)
                axes[midx+1][eidx].set(title="{} endmember: {}\n {} runs".format(methods[midx], eidx+1, n_runs))
                axes[midx+1][eidx].fill_between(freq_list, np.nanmin(endmember), np.nanmax(endmember),
                                                where=np.isnan(endmember), color='grey', alpha=0.5)

            plt.subplots_adjust(hspace=0.5)
            plt.savefig(Path(respath, dataset+'_endmembers.png'), dpi=200, format='png')
            plt.clf()

    if 'abundances' in metrics:

        cmap = sns.cubehelix_palette(start=.4, rot=-0.85, gamma=1.2, hue=2, light=0.85,
                                     dark=0.2, reverse=True, as_cmap=True)

        for dataset in datasets:

            n_bands, n_endmembers = results[dataset]['ref_endmembers'].shape
            fig = plt.figure(figsize=(7 * n_endmembers, 7 * len(methods) + 7 ))
            gs = fig.add_gridspec(len(methods) + 1, n_endmembers)

            # Plot reference abundances
            if 'ref_abundances' in results[dataset]:
                for n in range(n_endmembers):
                    ax = fig.add_subplot(gs[0, n], gid='endmember_ax_' + str(n))
                    sns.heatmap(results[dataset]['ref_abundances'][:, :, n], ax=ax, square=True, cmap=cmap)
                    ax.set(title="{},\nreference abundance map: {}".format(dataset, n + 1))

            # Plot abundances in their apropriate column

            for (midx, eidx) in product(range(len(methods)), range(n_endmembers)):
                if not m_sad[dataset][midx]:
                    for run in range(n_runs):
                        sad_m, idx_org_m, idx_hat_m, sad_k_m, s0 = \
                            calc_SAD_2(results[dataset]['ref_endmembers'],
                                       results[dataset][methods[midx]][run]['endmembers'])
                        m_sad[dataset][midx].append(sad_m)
                        m_idx_hat[dataset][midx].append(idx_hat_m)
                m_run = np.argmin(m_sad[dataset][midx])
                ax = fig.add_subplot(gs[midx + 1, eidx],
                                     gid=methods[midx] + '_ax_' + str(eidx) + str(m_run))
                sns.heatmap(results[dataset][methods[midx]][m_run]['abundances'][:, :,
                            m_idx_hat[dataset][midx][m_run][eidx]], ax=ax, square=True, cmap=cmap)
                ax.set(title="{} run: {}\nabundance map: {}".format(methods[midx], m_run, eidx+1))

            plt.subplots_adjust(hspace=0.6)
            plt.savefig(Path(respath, dataset+'_abundances.png'), dpi=200, format='png')
            plt.clf()

    print('Plots saved to', Path(respath).absolute())


def mse(Y, A, S):
    n, m = Y.shape
    e = np.sum(np.power(Y - np.matmul(A, S), 2)) / (n * m)
    return e


def improvement_only(a, b):
    try:
        if min(a) < b:
            return a + [min(a)]
        else:
            return a + [b]
    except ValueError:
        return a + [b]

def save_config(resdir, dataset_name, pars, best_value):
    config = ConfigParser()
    config.add_section(dataset_name)
    for key, value in pars.items():
        config[dataset_name][key.split('_')[-1]] = str(value)

    with open(Path(resdir, '{}_.cfg'.format(best_value)), 'w') as configfile:
        config.write(configfile)