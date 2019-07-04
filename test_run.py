from HySpUn import compare_methods

datapath='../../Datasets/'
datasets=['Samson', 'Urban4']
methods=['lhalf', 'ACCESSUnmixing','matlab_lhalf']
metrics=['endmembers', 'loss', 'SAD']

results = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics)
