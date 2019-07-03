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
        plot_results(results, metrics_to_plot, datasets, methods, num_runs, Path('./Results', timestamp))
        print('Plots saved to', Path('./Results', timestamp).absolute())

    with open(Path('./Results', timestamp, 'methods.cfg'), 'w') as configfile:
        config.write(configfile)

    return results, hsids


def plot_results(results, metrics, datasets, methods, num_runs, respath):
    sns.set_style('darkgrid')

    if 'loss' in metrics:
        fig, axes = plt.subplots(nrows=num_runs*len(datasets),
                                 ncols=len(methods),
                                 figsize=(5*num_runs*len(datasets),
                                          5*len(methods)))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('Loss', fontsize='xx-large', verticalalignment='baseline')
        for ax, (dset, run, method) in zip(axes.flatten(), product(datasets, range(num_runs), methods)):
            ax.plot(results[method][dset][run]['loss'])
            ax.set(title="{},\n{} run: {}".format(method, dset, run+1))

        plt.savefig(Path(respath, 'loss.png'), dpi=200, bbox_inches='tight', format='png')
        plt.clf()

    if 'SAD' in metrics:
        fig, axes = plt.subplots(nrows=num_runs*len(datasets),
                                 ncols=len(methods),
                                 figsize=(5*num_runs*len(datasets),
                                          5*len(methods)))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('SAD', fontsize='xx-large', verticalalignment='center')
        for ax, (dset, run, method) in zip(axes.flatten(), product(datasets, range(num_runs), methods)):
            ax.plot(results[method][dset][run]['SAD'])
            ax.set(title="{},\n{} run: {}".format(method, dset, run+1))

        plt.savefig(Path(respath, 'SAD.png'), dpi=200, bbox_inches='tight', format='png')
        plt.clf()

    if 'endmembers' in metrics:
        for dataset in datasets:
            n_bands, n_endmembers = results[dataset]['ref_endmembers'].shape
            fig, axes = plt.subplots(nrows=len(methods) + 1,
                                     ncols=n_endmembers,
                                     figsize=(5*len(methods),
                                              5*n_endmembers))
            fig.subplots_adjust(hspace=1)
            fig.suptitle('Endmembers',fontsize='xx-large', verticalalignment='center')
            ax = axes.flatten()
            for i in range(n_endmembers):
                ax[i].plot(results[dataset]['ref_endmembers'][:][i])
            for aidx, (method, eidx) in zip(range(len(axes.flatten()),n_endmembers+1), product(methods, range(n_endmembers))):
                for run in range(num_runs):
                    ax[aidx].plot(results[dataset][method][run]['endmembers'][:][eidx])
                ax[aidx].set(title="{},\n{} endmember: {}".format(method, dataset, eidx+1))

            plt.savefig(Path(respath, dataset+'_endmembers.png'), dpi=200, bbox_inches='tight', format='png')
            plt.clf()

    if 'abundances' in metrics:
        pass  # To be implemented
