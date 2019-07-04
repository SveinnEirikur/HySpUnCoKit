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

from tools.calc_SAD import calc_SAD_2

def compare_methods(datasets: list, methods: list, datapath: str = None, hsids: dict = None,
                    metrics_to_plot: list = None, plot: bool = True) -> dict:
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
            hsids[dataset] = HSID(datapath + dataset + '.mat', dataset_name=dataset,
                                  init_endmembers=np.load(Path(initials_dir, dataset, 'endmembers.npy')),
                                  init_abundances=np.load(Path(initials_dir, dataset, 'abundances.npy')))

    num_runs = config.getint('DEFAULT', 'num_runs', fallback=1)

    print('Running methods: ')
    print(*methods, sep=", ")
    print('On the following datasets: ')
    print(*datasets, sep=", ")
    print('Number of runs per method:', num_runs)

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

            output = plugin.run_method(hsids[dataset], respath, num_runs)
            results[dataset][method] = output

    for dataset in datasets:
        if hsids[dataset].ref_endmembers is not None:
            results[dataset]['ref_endmembers'] = hsids[dataset].ref_endmembers
        if hsids[dataset].ref_abundances is not None:
            results[dataset]['ref_abundances'] = hsids[dataset].ref_abundances

    if plot:
        plot_results(results, metrics_to_plot, datasets, methods,
                     num_runs, Path(config['DEFAULT']['output'], timestamp))

    with open(Path(config['DEFAULT']['output'], timestamp, 'methods.cfg'), 'w') as configfile:
        config.write(configfile)

    return results


def plot_results(results, metrics, datasets, methods, num_runs, respath):
    """

    :param results:
    :param metrics:
    :param datasets:
    :param methods:
    :param num_runs:
    :param respath:
    :return:
    """
    sns.set_style('darkgrid')

    if 'loss' in metrics:
        fig, axes = plt.subplots(nrows=num_runs*len(datasets),
                                 ncols=len(methods),
                                 figsize=(5*len(methods),
                                          5*num_runs*len(datasets)))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('Loss', fontsize='xx-large', verticalalignment='baseline')
        for ax, (dataset, run, method) in zip(axes.flatten(), product(datasets, range(num_runs), methods)):
            ax.plot(results[dataset][method][run]['loss'])
            ax.set(title="{},\n{} run: {}".format(method, dataset, run+1))

        plt.savefig(Path(respath, 'loss.png'), dpi=200, bbox_inches='tight', format='png')
        plt.clf()

    if 'SAD' in metrics:
        fig, axes = plt.subplots(nrows=num_runs*len(datasets),
                                 ncols=len(methods),
                                 figsize=(5*len(methods),
                                          5*num_runs*len(datasets)))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('SAD', fontsize='xx-large', verticalalignment='center')
        for ax, (dataset, run, method) in zip(axes.flatten(), product(datasets, range(num_runs), methods)):
            ax.plot(results[dataset][method][run]['SAD'])
            ax.set(title="{},\n{} run: {}".format(method, dataset, run+1))

        plt.savefig(Path(respath, 'SAD.png'), dpi=200, bbox_inches='tight', format='png')
        plt.clf()

    if 'endmembers' in metrics:
        for dataset in datasets:
            n_bands, n_endmembers = results[dataset]['ref_endmembers'].shape
            fig, axes = plt.subplots(nrows=len(methods) + 1,
                                     ncols=n_endmembers,
                                     figsize=(5*n_endmembers,
                                              5*len(methods)+5))
            fig.subplots_adjust(hspace=0.6)
            fig.suptitle('Endmembers', fontsize='xx-large', verticalalignment='center')
            ybound = [None for n in range(len(methods) + 1)]
            for n in range(n_endmembers):
                axes[0][n].plot(results[dataset]['ref_endmembers'][:, n])
                axes[0][n].set(title="{},\n{} endmember: {}".format(dataset, 'reference', n + 1))
                ybound[0] = axes[0][n].get_ybound()[1] if ybound[0] is None \
                    else max(axes[0][n].get_ybound()[1],ybound[0])
            for (midx, eidx) in product(range(len(methods)), range(n_endmembers)):
                for run in range(num_runs):
                    sad_m, idx_org_m, idx_hat_m, sad_k_m, s0 = \
                        calc_SAD_2(results[dataset]['ref_endmembers'],
                                   results[dataset][methods[midx]][run]['endmembers'])
                    axes[midx+1][eidx].plot(results[dataset][methods[midx]][run]['endmembers'][:, idx_hat_m[eidx]],
                                            label='Run: '+str(run))
                axes[midx+1][eidx].set(title="{} endmember: {}".format(methods[midx], eidx+1))
                axes[midx+1][eidx].legend(loc='best')
                ybound[midx+1] = axes[midx+1][n].get_ybound()[1] if ybound[midx+1] is None \
                    else max(axes[midx + 1][eidx].get_ybound()[1], ybound[midx+1])
            for midx in range(len(methods)+1):
                for eidx in range(n_endmembers):
                    axes[midx][eidx].set_ybound(upper=ybound[midx])

            plt.savefig(Path(respath, dataset+'_endmembers.png'), dpi=200, bbox_inches='tight', format='png')
            plt.clf()

    if 'abundances' in metrics:
        pass  # To be implemented

    print('Plots saved to', Path(respath).absolute())