from scipy import io as spio

def load_data(datapath):
    """ Load my .mat files.

    :param datapath:
    :return:
    """
    data = spio.loadmat(datapath, squeeze_me = True)

    n_row = data['lines']
    n_col = data['cols']

    Y = data['Y']

    S_ref = data.get('S_GT', None)
    A_ref = data['GT']

    n_band, n_end = A_ref.shape

    return Y, S_ref, A_ref, n_row, n_col, n_band, n_end
