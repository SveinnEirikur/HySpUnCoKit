# HySpUnCoKit

A tool for comparing Hyperspectral Unmixing methods.

## Installation

The repository can be cloned and the following requirements installed to your virtual environment of choice from `requirements.txt with
```bash
pip install -r requirements.txt 
```
or a comparable command. 
The code is compatible with Python versions 3.6 and 3.7.

#### Required packages

```
numpy==1.16.4
scipy==1.2.1
matplotlib
seaborn==0.9.0
tqdm==4.32.1
h5py==2.9.0
scikit-image==0.15.0
scikit-learn==0.21.2
tensorflow==1.13.1
keras_tqdm==2.0.1
dataclasses==0.6
```

Apart from the `dataclasses` package all packages are available from the Anaconda package manager as well as PyPI.
The `dataclasses`package is only necessary for Python 3.6 as it is a backport of Python 3.7 functionality.

The `tensorflow` and `keras_tqdm` are only necessary if working with machine learning methods.

## Adding methods

Methods can be added by creating a python wrapper script named `run_as_plugin.py` with a module `run_method(data, resdir, num_runs)`
that accepts Hyperspectral Image data passed using the provided HSID class, a path to a directory where it should output any resulting files and the number of times the unmixing should be repeated.

The wrapper script should contain any necessary parameters and code to use the unmixing method it provides or be able to load them from a cfg file as the provided examples do.

The `run_method` module should return results in the form of a `list` of `dict` objects where each `dict`contains the outcome of one run.
It should contain any metrics or results you wish to access.
```python
results = [{'endmembers': endmembers, 'abundances': abundances, 'loss': history['loss'], 'SAD': history['SAD']}]
```

The provided example methods also save a `.mat` file with the results of each run in case of runtime interruptions, for further processing and archival purposes.

### Matlab methods

To use methods implemented in Matlab the Matlab Python Engine must be installed to your virtual environment following the instructions provided on the Mathworks website and comparable wrapper scripts created. 
See the `matlabplugins/lhalf/run_as_plugin.py` code as an example of such.

## Adding data

Hyper Spectral image data can be passed to the method using the provided HSID class, either by filling in the data manually or loading from a suitably formated `.mat` file:

```python
hsids = [HSID(dataset_name='Samson', data=data_nparray, init_endmembers=endmember_nparray), 
         HSID('/path/to/dataset.mat', init_endmembers=endmember_nparray)]
```

Numpy arrays are used for the hyperspectral image data itself.

The expected structure of the `.mat` files has the following variable names:
* `Y`: The hyperspectral image data to be unmixed, as a 2D array shaped as (bands, pixels)
* `lines`: The number of image pixels per line
* `cols`: The number of image pixels per column
* `GT`: Reference endmembers ("Ground truth" endmembers)
* `S_GT`: Reference abundances ("Ground truth" abundance maps)

HSID objects can however be initialized with different variable names and with reference data from a separate `.mat` file by initializing them manually.
