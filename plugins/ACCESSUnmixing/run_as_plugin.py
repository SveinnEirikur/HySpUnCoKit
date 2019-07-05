from tqdm import tqdm  # _notebook as tqdm
from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
import os

from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras import backend as K

if __package__ == "ACCESSUnmixing":
    from ACCESSUnmixing.ACCESSUnmixing import Autoencoder
    from ACCESSUnmixing.unmixing.HSI import HSI
    from ACCESSUnmixing.unmixing.losses import SAD
else:
    from ACCESSUnmixing import Autoencoder
    from unmixing.HSI import HSI
    from unmixing.losses import SAD


def run_method(data, resdir, num_runs):
    dataset = data.dataset_name
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__)))
    configpath = os.path.join(__location__, 'datasets.cfg')
    config = ConfigParser()
    config.read(configpath)
    if not config.has_section(dataset):
        warn('No settings found for ' + dataset + ', using defaults.')
        config.add_section(dataset)

    datapath = data.data_path
    init_endmembers = data.init_endmembers
    activation = LeakyReLU(0.2)

    opt_name = config.get(dataset, 'optimizer')
    opt_cfg = dict(config._sections[dataset + ' ' + opt_name])
    for key, value in opt_cfg.items():
        opt_cfg[key] = float(value)
    optimizer = { 'class_name': opt_name, 'config': opt_cfg }
    l2 = config.getfloat(dataset, 'l2')
    l1 = config.getfloat(dataset, 'l1')
    num_patches = config.getint(dataset, 'num_patches')
    epochs = config.getint(dataset, 'epochs')
    batch_size = config.getint(dataset, 'batch_size')
    plot_every_n = config.getint(dataset, 'plot_every_n')
    n_band, n_end = init_endmembers.shape

    my_data = HSI(datapath)
    results = []

    for i in tqdm(range(num_runs), desc="Runs", unit="runs"):
        my_data.load_data(normalize=True, shuffle=False)
        unmixer = Autoencoder(n_end=n_end, data=my_data, activation=activation,
                              optimizer=optimizer, l2=l2, l1=l1, plot_every_n = plot_every_n)
        unmixer.create_model(SAD)
        my_data.make_patches(1, num_patches=num_patches, use_orig=True)
        history = unmixer.fit(epochs=epochs, batch_size=batch_size)
        resfile = 'Run_' + str(i + 1) + '.mat'
        endmembers = unmixer.get_endmembers().transpose()
        abundances = unmixer.get_abundances().reshape(data.n_rows,data.n_cols,data.n_endmembers).transpose((1, 0, 2))
        resdict = {'endmembers': endmembers,
                   'abundances': abundances,
                   'loss': history.history['loss'],
                   'SAD': history.history['SAD']}
        results.append(resdict)
        spio.savemat(resdir / resfile, results[i])
        del unmixer
        K.clear_session()
    return results
