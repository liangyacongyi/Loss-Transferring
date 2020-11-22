#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_aug.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/3/15 11:28 AM   liangcong      1.0   augmentation for datasets
"""
import numpy as np
import random


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _flip_leftright(batch):
    for i in range(len(batch)):
        batch[i] = np.fliplr(batch[i])
    return batch


def torch_input(batch):
    batch = batch.reshape([-1, 32, 32, 3])
    batch = _random_crop(batch, [32, 32, 3], padding=4)
    batch = _random_flip_leftright(batch)
    return np.array(batch).reshape(-1, 32 * 32 * 3)
