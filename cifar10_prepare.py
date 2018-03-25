# coding=utf-8
# use python2.7 to run this file
from __future__ import print_function
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)



import os
import numpy as np
import tarfile
import h5py

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def prepare_h5py(train_image, train_label, test_image, test_label, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    data_id = open(os.path.join(data_dir,'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i%(image.shape[0]/100)==0:
            bar.update(i/(image.shape[0]/100))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(10)
        label_vec[label[i]%10] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return

download_path = './datasets'
data_dir = os.path.join(download_path, 'cifar10')
k = 'cifar-10-python.tar.gz'
target_path = os.path.join(data_dir, k)


tarfile.open(target_path, 'r:gz').extractall(data_dir)

num_cifar_train = 50000
num_cifar_test = 10000

target_path = os.path.join(data_dir, 'cifar-10-batches-py')
train_image = []
train_label = []
for i in range(5):
    fd = os.path.join(target_path, 'data_batch_'+str(i+1))
    dict = unpickle(fd)
    train_image.append(dict['data'])
    train_label.append(dict['labels'])

train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32*32*3])
train_label = np.reshape(np.array(np.stack(train_label, axis=0)), [num_cifar_train])

fd = os.path.join(target_path, 'test_batch')
dict = unpickle(fd)
test_image = np.reshape(dict['data'], [num_cifar_test, 32*32*3])
test_label = np.reshape(dict['labels'], [num_cifar_test])

prepare_h5py(train_image, train_label, test_image, test_label, data_dir, [32, 32, 3])
