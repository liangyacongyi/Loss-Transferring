#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   resnet_cifar_model.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/3/15 09:10 AM   liangcong    1.0    resnet model for cifar_10/100
                                         in paper "Deep Residual Learning for Image Recognition"
"""
import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 1
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training)


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, fg=False):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=fg,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
        # kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
        data_format=data_format)


def _building_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=1, data_format=data_format, fg=False)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides, data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=strides, data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=1, data_format=data_format)
    return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters,
                                  kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides, data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=1, strides=1, data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=strides, data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters,
                                  kernel_size=1, strides=1, data_format=data_format)
    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, training, name, data_format):
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1,
                                    strides=strides, data_format=data_format)

    inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


def model(cifar, n, bottleneck=False, version=1, filters=16):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _DATA_FORMAT = 'channels_last'
    block_sizes = [n, n, n]  # n=3(20 layers), n=5(32 layers), n=3(20 layers),
    block_strides = [1, 2, 2]

    if bottleneck:
        if version == 1:
            block_fn = _bottleneck_block_v1
        else:
            block_fn = _bottleneck_block_v2
    else:
        if version == 1:
            block_fn = _building_block_v1
        else:
            block_fn = _building_block_v2

    if cifar == 10:
        _NUM_CLASSES = 10
    else:
        _NUM_CLASSES = 100

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        count_label = tf.placeholder(tf.int32, shape=[_NUM_CLASSES])
        training = tf.placeholder(tf.bool, name='training')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    # conv1 layer
    with tf.name_scope('conv1'):
        inputs = conv2d_fixed_padding(inputs=x_image, filters=16, kernel_size=3, strides=1, data_format=_DATA_FORMAT)

    if version == 1:
        inputs = batch_norm(inputs, training, _DATA_FORMAT)
        inputs = tf.nn.relu(inputs)

    with tf.name_scope('residual_part'):
        for i, num_blocks in enumerate(block_sizes):
            num_filters = filters * (2**i)
            inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=bottleneck,
                                 block_fn=block_fn, blocks=num_blocks, strides=block_strides[i],
                                 training=training, name='block_layer{}'.format(i + 1), data_format=_DATA_FORMAT)

    axes = [2, 3] if _DATA_FORMAT == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keep_dims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')

    inputs = tf.reshape(inputs, [-1, 64])
    output = tf.layers.dense(inputs=inputs, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                             units=_NUM_CLASSES)
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(output, dimension=1)
    return x, y, output, global_step, y_pred_cls, count_label, training
