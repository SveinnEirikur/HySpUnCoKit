import run_as_plugin
import numpy as np
from pathlib import Path


init_endmem = np.load('../../initials/Samson/endmembers.npy')
init_abundances = np.load('../initials/Samson/abundances.npy')
hsidata = np.load('../initials/Samson/HSI.npy')
run_as_plugin.run_method(hsidata, resdir, num_runs)
