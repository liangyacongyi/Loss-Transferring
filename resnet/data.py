#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/3/15 11:28 AM   liangcong      1.0   download or load cifar_10/100 dataset
"""
import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import data_aug as data_aug


def get_data_set(name="train", cifar=10, aug=False):
    """
    Get the dataset cifar10/cifar100
    :param name: train or test
    :param cifar: 10 or 100. 10 means cifar10, 100 means cifar100
    :param aug:  True or False. True means the dataset is augmented, False means not
    :param gcn:  True or False. True means the dataset is processed by gcn, False means not
    :param whitten: True or False. True means the dataset is processed by zca, False means not
    :return: the data, label and description of label
    """
    x = None
    y = None
    l = None

    maybe_download_and_extract(cifar)  # download the dataset if not exist

    if cifar == 10:
        folder_name = "cifar_10"
        f = open('/root/data/lc/data/data_set/' + folder_name + '/batches.meta', 'rb')
        label_str = "label_names"
    else:
        folder_name = "cifar_100"
        f = open('/root/data/lc/data/data_set/' + folder_name + '/meta', 'rb')
        label_str = "fine_label_names"
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict[label_str]

    if name is "train":
        if cifar == 10:
            for i in range(5):
                f = open('/root/data/lc/data/data_set/' + folder_name + '/data_batch_' + str(i + 1), 'rb')
                datadict = pickle.load(f, encoding='latin1')
                f.close()
                _X = datadict["data"]
                _Y = datadict['labels']
                _X = np.array(_X)
                _X = _X.reshape([-1, 3, 32, 32])
                _X = _X.transpose([0, 2, 3, 1])

                if (aug):
                    cropped_data = data_aug._random_crop(_X, [32, 32, 3], padding=4)
                    flipped_data = data_aug._flip_leftright(_X)
                    _X = np.concatenate((_X, cropped_data, flipped_data), axis=0)
                    _Y = np.concatenate((_Y, _Y, _Y))
                if x is None:
                    x = _X
                    y = _Y
                else:
                    x = np.concatenate((x, _X), axis=0)
                    y = np.concatenate((y, _Y), axis=0)
            f_x = x.reshape(-1, 32 * 32 * 3)
            f_y = np.array(y)
        else:
            f = open('/root/data/lc/data/data_set/' + folder_name + '/train', 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()
            _X = datadict["data"]
            _Y = datadict['fine_labels']
            _X = np.array(_X)
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            if (aug):
                print ("start aug")
                cropped_data = data_aug._random_crop(_X, [32, 32, 3], padding=4)
                #flipped_data_1 = data_aug._flip_leftright(_X)
                #flipped_data_2 = data_aug._flip_leftright(cropped_data)
                flipped_data = data_aug._random_flip_leftright(_X)
                #_X = np.concatenate((_X, cropped_data, flipped_data_1, flipped_data_2), axis=0)
                #_Y = np.concatenate((_Y, _Y, _Y, _Y))
                _X = np.concatenate((_X, cropped_data, flipped_data), axis=0)
                _Y = np.concatenate((_Y, _Y, _Y))
            x = _X
            y = _Y
            f_x = x.reshape(-1, 32 * 32 * 3)
            f_y = np.array(y)

    elif name is "test":
        if cifar == 10:
            f = open('/root/data/lc/data/data_set/' + folder_name + '/test_batch', 'rb')
            label_str = "labels"
        else:
            f = open('/root/data/lc/data/data_set/' + folder_name + '/test', 'rb')
            label_str = "fine_labels"
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = datadict[label_str]

        x = np.array(x)
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        f_x = x.reshape(-1, 32 * 32 * 3)
        f_y = np.array(y)

    def dense_to_one_hot(labels_dense, cifar):
        if cifar == 10:
            num_classes = 10
        else:
            num_classes = 100
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    return f_x, dense_to_one_hot(f_y, cifar), l


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract(cifar):
    """
    Download and extract different dataset
    :param cifar: Dataset name, (cifar_10, cifar_100)
    :return: none
    """
    main_directory = "/root/data/lc/data/data_set/"
    data_directory = main_directory + "cifar_" + str(cifar) + "/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    if cifar == 10 and not os.path.exists(data_directory):
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print("Cifar10 dataset download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")
        os.rename(main_directory + "./cifar-10-batches-py", data_directory)
        os.remove(zip_cifar_10)
    elif cifar == 100 and not os.path.exists(data_directory):
        url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print("Cifar100 dataset download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")
        os.rename(main_directory + "./cifar-100-python", data_directory)
        os.remove(zip_cifar_100)



