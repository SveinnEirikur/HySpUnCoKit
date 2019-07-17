from HySpUn import compare_methods

datapath='../../Datasets/'
datasets=['Samson']
methods=['lhalf']
metrics=['endmembers', 'abundances']

results = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics)
