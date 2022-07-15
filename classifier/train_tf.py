# -*- coding:utf-8 -*-
"""
project: MaskCycleGANproject_time
file: train_tf.py
author: Jiang Xiangyu
create date: 7/3/2022 2:22 PM
description: None
"""
# HAR classification
# Author: Burak Himmetoglu
# 8/15/2017
import pickle

import matplotlib as mpl
from tensorflow.python.keras import losses
from tensorflow.python.keras.optimizer_v2.adam import Adam
from model.Resnet_tf import Resnet
mpl.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
import numpy as np

np.random.seed(1337)  # for reproducibility


def read_data(datadir, split="train"):
    datapath = os.path.join(datadir, split,f'{split}_signal.pickle')
    with open(datapath,'rb') as datafile:
        path_signals = pickle.load(datafile)
    labelpath = os.path.join(datadir, split,f'y_{split}.pickle')
    with open(labelpath,'rb') as labelfile:
        labels = pickle.load(labelfile)
    # Read time_expand-series data
    n_channels = 1
    # Initiate array
    x = np.zeros((len(labels), 128,128, n_channels))
    i_ch = 0
    x[:, :,:, i_ch] = path_signals
    # Return
    return x, labels.squeeze()


def standardization(train, test):
    mean=np.mean(np.vstack((train,test)),axis=0)
    std = np.std(np.vstack((train, test)), axis=0)
    x_data_train = (train - mean[None, :, :]) / std[None, :, :]
    x_data_test = (test - mean[None, :, :]) / std[None, :, :]
    return x_data_train, x_data_test


def one_hot(labels, n_class=3):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

X_train, Y_train = read_data(datadir='dataset', split="train")  # train
X_test, labels_test = read_data(datadir='dataset',  split="test")  # test
# X_train, X_test=standardization(X_train, X_test)
# X_train, X_vld, labels_tr, labels_vld = train_test_split(X_tr, labels_tr, test_size=0.001)
# One-hot encoding:
Y_train = one_hot(Y_train)
# Y_vld = one_hot(labels_vld)
Y_test = one_hot(labels_test)

model = Resnet([2,2,2,2],3)
model.build(input_shape=(None,128,128,1))
model.compile(optimizer=Adam(1e-4,),
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
output = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_test, Y_test),
        callbacks=[],
        verbose=1,
        workers=4
    )
result = model.evaluate(X_test, Y_test)
print("loss=%.2f" % result[0])
print("acc=%.2f" % result[1])
model.save_weights(f'resnet_model_{result[1]:.2f}.h5')
