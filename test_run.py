from HySpUn import compare_methods

def test_run(datapath='../../Datasets/',
             datasets=['Samson'],
             methods=['lhalf'],
             metrics=['endmembers']):
    results, hsids = compare_methods(datasets, methods, datapath=datapath, metrics_to_plot=metrics)
    return results, hsids
