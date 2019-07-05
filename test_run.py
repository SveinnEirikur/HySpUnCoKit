from HySpUn import compare_methods

datapath='../../Datasets/'
datasets=['Samson','Urban4']
methods=['ACCESSUnmixing','lhalf','matlab_lhalf']
metrics=['endmembers', 'abundances','loss','SAD']

results = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics)
