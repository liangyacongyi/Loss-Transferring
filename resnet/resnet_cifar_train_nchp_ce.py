#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   resnet_cifar_train_nchp_ce.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/12 11:28 AM   liangcong      1.0   Training ResNet model with CIFAR-10/100

run this file with
"python resnet_cifar_train_nchp_ce.py --num_blocks=3 --data_type=cifar_10"
"""
import numpy as np
import tensorflow as tf
import datetime
import os

from sklearn.metrics import confusion_matrix
from data import get_data_set
from data_aug import torch_input
from resnet_cifar import model
tf.set_random_seed(1234)
np.random.seed(1234)
# ------------------------------command line argument---------------------------------
tf.app.flags.DEFINE_float("f_p", 1.0, "fixed point")
tf.app.flags.DEFINE_string("data_type", 'cifar_10', "cifar_10")
tf.app.flags.DEFINE_float("weight_decay", 0.00035, "weight decay, 0.00035")
tf.app.flags.DEFINE_integer("num_blocks", 3, "layers number: 6*num_blocks+2")

FLAGS = tf.app.flags.FLAGS
_FIXED_POINT = FLAGS.f_p
_DATA_TYPE = FLAGS.data_type
_WEIGHT_DECAY = FLAGS.weight_decay
_NUM_BLOCKS = FLAGS.num_blocks
# ------------------------------------------------------------------------------------

# ------------------------------command line argument judge---------------------------
if not isinstance(_FIXED_POINT, float):
    print("f_p must be a float number")
    os._exit(0)
if not isinstance(_DATA_TYPE, str):
    print("data_type must be a string")
    os._exit(0)
if not isinstance(_NUM_BLOCKS, int):
    print("num_blocks must be a int number")
    os._exit(0)

if _DATA_TYPE not in ('cifar_10', 'cifar_100'):
    print("_DATA_TYPE must be cifar_10 or cifar_100")
    os._exit(0)
if _NUM_BLOCKS not in (3, 5, 7, 9, 18):
    print("_DATA_TYPE must be selected from {3,5,7,9,18}")
    os._exit(0)
# ------------------------------------------------------------------------------------

# ------------------------------fixed parameters--------------------------------------
_ITERATION = 96000
_BATCH_SIZE = 128
_CLASS_SIZE = int(_DATA_TYPE.split('_')[1])
_loss_style = "nchp_ce"
# ------------------------------------------------------------------------------------

print("num_layers: %d, data_type: %s" % (6 * FLAGS.num_blocks + 2, _DATA_TYPE))

# ------------------------------create relative folders-------------------------------
PAR_PATH = "./" + _DATA_TYPE + "/" + str(6 * _NUM_BLOCKS + 2) + "/" + _loss_style + "/FP_" + str(_FIXED_POINT)
_MODEL_SAVE_PATH = os.path.join(PAR_PATH, "model/")

if not os.path.exists(PAR_PATH):
    os.makedirs(PAR_PATH)
if not os.path.exists(_MODEL_SAVE_PATH):
    os.makedirs(_MODEL_SAVE_PATH)
_TENSORBOARD_SAVE_PATH = os.path.join(PAR_PATH, "tensorboard")
# ------------------------------------------------------------------------------------

x, y, output, global_step, y_pred_cls, c, phase_train = model(_CLASS_SIZE, _NUM_BLOCKS)

# ------------------------------data pre-process-------------------------------------
train_x, train_y, train_l = get_data_set(name="train", cifar=_CLASS_SIZE)
test_x, test_y, test_l = get_data_set(name="test", cifar=_CLASS_SIZE)

print("mean subtracted")
mean = np.mean(train_x, axis=1)
train_x = train_x - mean[:, np.newaxis]
test_mean = np.mean(test_x, axis=1)
test_x = test_x - test_mean[:, np.newaxis]
print("mean subtracted end")
epoch_size = len(train_x)
if epoch_size % _BATCH_SIZE == 0:
    steps_per_epoch = epoch_size / _BATCH_SIZE
else:
    steps_per_epoch = int(epoch_size / _BATCH_SIZE) + 1
# ------------------------------------------------------------------------------------
alpha1_boundaries = [400, 32000]
alpha1_value = [0.01, 1., 0.]
alpha1 = tf.train.piecewise_constant(global_step, alpha1_boundaries, alpha1_value)

alpha2_boundaries = [400, 32000]
alpha2_value = [1., 0., 1.]
alpha2 = tf.train.piecewise_constant(global_step, alpha2_boundaries, alpha2_value)

n_chp_loss = tf.reduce_sum(tf.square(output-_FIXED_POINT*y)) / (2 * _BATCH_SIZE)
ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
loss = alpha1 * n_chp_loss + alpha2 * ce_loss
# ------------------------------------------------------------------------------------

# ------------------------------weight regularization---------------------------------
if _CLASS_SIZE == 10:
    wd_boundaries = [32000]
    wd_value = [_WEIGHT_DECAY, _WEIGHT_DECAY]
    wd = tf.train.piecewise_constant(global_step, wd_boundaries, wd_value)
else:
    wd_boundaries = [32000]
    wd_value = [0.0002, _WEIGHT_DECAY]
    wd = tf.train.piecewise_constant(global_step, wd_boundaries, wd_value)
t_v = tf.losses.get_regularization_losses()
w12 = tf.add_n([t_v[i] for i in range(len(t_v))])
wl2_loss = w12 * wd
# ------------------------------------------------------------------------------------

# ------------------------------optimal algorithm setting-----------------------------
if _CLASS_SIZE == 10:
    boundaries = [32000, 64000, 80000]  # for layer-110
    values = [0.1, 0.3, 0.03, 0.003]
else:
    boundaries = [32000, 64000, 80000]
    values = [0.1, 0.2, 0.02, 0.002]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, name='Momentum1', use_nesterov=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = train_op.minimize(loss + wl2_loss, global_step=global_step)
# ------------------------------------------------------------------------------------

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)
tf.summary.scalar("Loss", loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
train_writer = tf.summary.FileWriter(_TENSORBOARD_SAVE_PATH, sess.graph)
sess.run(tf.global_variables_initializer())


def train(itera, f):
    """
    Train CNN
    :param itera: numbers of iteration
    :param f: file for storing the training log
    :return:
    """
    global train_x
    global train_y
    
    _ce_loss = []
    _n_chp_loss = []
    _test_acc = []
    _cm = []
    _feature = []
    
    print('start training')
    print('start training', file=f)

    print("parameter settings, num_layers: %d, data_type: %s, loss_style: %s, f_p: %f"
          % (6 * FLAGS.num_blocks + 2, _DATA_TYPE, _loss_style, _FIXED_POINT))
    print("parameter settings, num_layers: %d, data_type: %s, loss_style: %s, f_p: %f"
          % (6 * FLAGS.num_blocks + 2, _DATA_TYPE, _loss_style, _FIXED_POINT), file=f)
    
    for i in range(itera):
        randidx = np.random.choice(len(train_x), size=_BATCH_SIZE, replace=False)
        batch_xs = torch_input(train_x[randidx])
        batch_ys = train_y[randidx]
            
        i_global, _, l_loss, l_ce_loss, l_n_chp_loss, l_acc = \
            sess.run([global_step, optimizer, loss, ce_loss, n_chp_loss, accuracy],
                     feed_dict={x: batch_xs, y: batch_ys, phase_train: True})

        if i_global % 10 == 0:
            print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, n_chp_loss: %.4f"
                  % (i_global, l_acc, l_loss, l_ce_loss, l_n_chp_loss))
            print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, n_chp_loss: %.4f"
                  % (i_global, l_acc, l_loss, l_ce_loss, l_n_chp_loss), file=f)

        if i_global % 1000 == 0 or i_global == itera:
            data_merged = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, phase_train: True})
            print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, n_chp_loss: %.4f"
                  % (i_global, l_acc, l_loss, l_ce_loss, l_n_chp_loss))
            print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, n_chp_loss: %.4f"
                  % (i_global, l_acc, l_loss, l_ce_loss, l_n_chp_loss), file=f)

            acc, cm, feature = predict_test(f)

            _test_acc.append(acc)
            _ce_loss.append(l_ce_loss)
            _n_chp_loss.append(l_n_chp_loss)

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy/test", simple_value=acc)])
            train_writer.add_summary(data_merged, i_global)
            train_writer.add_summary(summary, i_global)
                
            if i_global == 1000:
                saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")
                print("Saved checkpoint.", file=f)
                _temp_acc = acc
                _temp_cm = cm
                _temp_feature = feature
            
            if acc > _temp_acc:
                saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")
                print("Saved checkpoint.", file=f)
                _temp_acc = acc
                _temp_cm = cm
                _temp_feature = feature
    
    _feature.append(_temp_feature)
    _cm.append(cm)
    
    return _ce_loss, _n_chp_loss, _test_acc, _cm, _feature


def predict_test(f, show_confusion_matrix=False):
    """
    Make prediction for all images in test_x
    :param show_confusion_matrix: default false
    :return: accuracy
    """
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    feature = np.zeros(shape=(len(test_x), _CLASS_SIZE), dtype=np.float)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j], feature[i:j, :] = \
            sess.run([y_pred_cls, output], feed_dict={x: batch_xs, y: batch_ys, phase_train: False})
        i = j
    
    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)), file=f)
    
    cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))
    
    return acc, cm, feature


def main(_):
    f = open(os.path.join(PAR_PATH, "train_info_.txt"), "w")
    time1 = datetime.datetime.now()
    ce_loss, n_chp_loss, test_acc, cm, feature = train(_ITERATION, f)
    time2 = datetime.datetime.now()
    duration = time2 - time1
    print("duration:", duration)
    print("duration:", duration, file=f)
    print("the best test_acc: %.3f, in epoch %d" % (np.max(test_acc), np.argmax(test_acc)))
    print("the best test_acc: %.3f, in epoch %d" % (np.max(test_acc), np.argmax(test_acc)), file=f)
    f.close()
    
    np.savetxt(os.path.join(PAR_PATH, "ce_loss.txt"), ce_loss)
    np.savetxt(os.path.join(PAR_PATH, "test_acc.txt"), test_acc)
    np.savetxt(os.path.join(PAR_PATH, "n_chp_loss.txt"), n_chp_loss)
    np.save(os.path.join(PAR_PATH, "cm.npy"), cm)
    np.save(os.path.join(PAR_PATH, "feature.npy"), feature)


if __name__ == "__main__":
    tf.app.run()

sess.close()

