
params = dict()
params['batch_size']=4
params['epochs'] = 100
params['num_workers']=1
params['momentum']=0.9
params['learning_rate']=0.001
params['weight_decay']=1e-4
params['gpu']=0
params['key_savepath'] = '..'
params['num_savepath'] = '..'
params['dataset_savepath'] = '/mnt/data/guest0/sudoku_dataset'
params['csv_savepath'] = '.'
params['mnist_path'] = '/mnt/data/guest0/sudoku_dataset/mnist_png/training'
params['background'] = '/mnt/data/guest0/sudoku_dataset/newspapers/data'
params['key_model'] = './key.pth'
params['num_model'] = './num.pth'
