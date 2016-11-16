import os
import numpy as np
from scipy import misc


def cifar_data(size='1000', dataset='train', path='.'):
    if size == '1000':
        path = path + '/cifar_data_1000_100/'
        if dataset == 'train':
            path = path + 'train/'
        elif dataset == 'test':
            path = path + 'test/'
    elif size == '100':
        path = path + '/cifar_data_100_100/'
        if dataset == 'train':
            path = path + 'train/'
        elif dataset == 'test':
            path = path + 'test/'

    files_list = os.listdir(path)
    n_samples = len(files_list)
    X = np.ndarray((n_samples, 3072))
    y = np.ndarray((n_samples, 10))

    for i in range(n_samples):
        class_label = files_list[i].split('_')[0]
        img_arr = np.asarray(misc.imread(path + files_list[i])).ravel()
        X[i] = img_arr

        y[i] = np.eye(10)[int(class_label)]

    return X, y


X, y = cifar_data(size='1000', dataset='train', path='.')
print X
print y
