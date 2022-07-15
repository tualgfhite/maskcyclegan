# HAR classification
# Author: Burak Himmetoglu
# 8/15/2017
import pickle

import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import regularizers, losses
from tensorflow.python.keras.optimizer_v2.adam import Adam

mpl.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.models import Sequential
from scipy import io
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


def get_batches(x, y, batch_size=100):
    """ Return a generator for batches """
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(x), batch_size):
        yield x[b:b + batch_size], y[b:b + batch_size]


# TIME_STEPS = 128     # same as the height of the image
# INPUT_SIZE = 128     # same as the width of the image
BATCH_SIZE = 128# 32-88 64-87 128-90  256-88     (32-88 64-91.8 256-86)
BATCH_INDEX = 0
OUTPUT_SIZE = 3
CELL_SIZE = 50
LR = 0.0001#0.001
TRAIN_VALIDATION_RATIO = 0.9
epochs = 500  #1000 这个决定训练量大小，现在是训练10000*20=20万个数据，假如总数据量是4万个，那么将把所有数据训练5次
# 注意这里的epochs和plot.py里的epoch应该大小一致

# ## 注意路径中不能包含硬盘名字，如 C: 、 F: ，必须以硬盘名字之后的目标文件开头，且路径首末要有 “ / ” ####
X_tr, labels_tr = read_data(datadir='dataset', split="train")  # train
#X_ex, labels_ex = read_data(data_path="lofar_data/", split="expanded")  # expanded
X_test, labels_test = read_data(datadir='dataset',  split="test")  # test

# Train/Validation Split
# X_train, X_vld, labels_train, labels_vld = train_test_split(X_tr, labels_tr,test_size=1/9,stratify = labels_train,
# random_state = 123)
X_tr,X_test=standardization(X_tr.reshape([-1,1,128,128]),X_test.reshape([-1,1,128,128]))

X_train, X_vld, labels_train, labels_vld = train_test_split(X_tr, labels_tr, test_size=0.001)
# One-hot encoding:
Y_train = one_hot(labels_train)
Y_vld = one_hot(labels_vld)
Y_test = one_hot(labels_test)

model = Sequential()
model.add(Convolution2D(
    batch_input_shape=(None,1,128, 128),
    filters=4,
    kernel_size=5,
    strides=1,
    padding='same',
    kernel_regularizer=regularizers.l2(0.0001),
    activation='relu'
))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
model.add(Convolution2D(8, 5, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(2, 2, 'same'))
model.add(Convolution2D(16, 5, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(2, 2, 'same'))
model.add(Convolution2D(32, 5, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(2, 2, 'same'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss=losses.BinaryCrossentropy(label_smoothing=0.05),
              metrics=['accuracy'])
output = model.fit(
        X_train, Y_train,
        epochs=600,
        batch_size=256,
        validation_data=(X_test, Y_test),
        callbacks=[],
        verbose=1
    )
result = model.evaluate(X_test, Y_test)
print("loss=%.2f" % result[0])
print("acc=%.2f" % result[1])
model.save('cnn_lofar_model.h5')