# HAR classification
# Author: Burak Himmetoglu
# 8/15/2017
import matplotlib as mpl
from tensorflow.python.keras.optimizer_v2.adam import Adam

mpl.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
from scipy import io
np.random.seed(1337)  # for reproducibility
import pickle


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




def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    x_data_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    x_data_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]

    return x_data_train, x_data_test


def one_hot(labels, n_class=4):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def standardization(train, test):
    mean=np.mean(np.vstack((train,test)),axis=0)
    std = np.std(np.vstack((train, test)), axis=0)
    x_data_train = (train - mean[None, :, :]) / std[None, :, :]
    x_data_test = (test - mean[None, :, :]) / std[None, :, :]
    return x_data_train, x_data_test

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
OUTPUT_SIZE = 4
CELL_SIZE = 50
LR = 0.0001#0.001
TRAIN_VALIDATION_RATIO = 0.9999
epochs = 2000  #1000 这个决定训练量大小，现在是训练10000*20=20万个数据，假如总数据量是4万个，那么将把所有数据训练5次
# 注意这里的epochs和plot.py里的epoch应该大小一致

# ## 注意路径中不能包含硬盘名字，如 C: 、 F: ，必须以硬盘名字之后的目标文件开头，且路径首末要有 “ / ” ####
X_train, Y_train = read_data(datadir='dataset', split="train")  # train
X_test, labels_test = read_data(datadir='dataset',  split="test")  # test
X_train, X_test=standardization(X_train, X_test)
# X_train, X_vld, labels_tr, labels_vld = train_test_split(X_tr, labels_tr, test_size=0.001)
# One-hot encoding:
Y_train = one_hot(Y_train)
# Y_vld = one_hot(labels_vld)
Y_test = one_hot(labels_test)



#拼接expanded数据和原始train数据
#X_train = np.vstack((X_train, X_ex))
#Y_train = np.vstack((Y_train, Y_ex))
#print(Y_train.size)

# build  model
model = Sequential()

# build CNN model
# Conv layer 1 output shape (4, 16384)
model.add(Convolution2D(
    batch_input_shape=(None,  128, 128,1),
    filters=4,
    kernel_size=5,
    strides=1,
    padding='same'# Padding method
    # data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (4, 4096)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',  # Padding method
    # data_format='channels_first',
))

# Conv layer 2 output shape (8, 4096)
model.add(Convolution2D(8, 5, strides=1, padding='same'))  # , data_format='channels_first'))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# Pooling layer 2 (max pooling) output shape (8, 1024)
model.add(MaxPooling2D(2, 2, 'same'))  # , data_format='channels_first'))


# Conv layer 3 output shape (16, 1024)
model.add(Convolution2D(16, 5, strides=1, padding='same'))  # , data_format='channels_first'))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# Pooling layer 3 (max pooling) output shape (16, 256)
model.add(MaxPooling2D(2, 2, 'same'))  # , data_format='channels_first'))


# Conv layer 3 output shape (32, 256)
model.add(Convolution2D(32, 5, strides=1, padding='same'))  # , data_format='channels_first'))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D())
# Pooling layer 3 (max pooling) output shape (32, 64)
model.add(MaxPooling2D(2, 2, 'same'))  # , data_format='channels_first'))


# build RNN model
# RNN cell
# model.add(SimpleRNN(
#     # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
#     # Otherwise, model.evaluate() will get error.
#     batch_input_shape=(None, 32, 64),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
#     output_dim=CELL_SIZE,
#     unroll=True,
# ))
# model.add(LSTM(126))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
train_acc = []
train_loss = []

real_train_acc = []
real_train_loss = []
i = 0
for step in range(epochs + 1):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = Y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]

    train_cost, train_accuracy = model.train_on_batch(X_batch, Y_batch)
    real_train_acc.append(train_accuracy)
    real_train_loss.append(train_cost)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 50 == 0:
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
        print('train cost: ', cost, 'train accuracy: ', accuracy)
        train_acc.append(accuracy)
        train_loss.append(cost)

model.save('cnn_lofar_model.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
# del model  # deletes the existing model

# save the train data
if not os.path.exists('CNN_lofar_train_acc_loss'):
    os.mkdir('CNN_lofar_train_acc_loss')
io.savemat('CNN_lofar_train_acc_loss/' + str(epochs) + 'train_acc_CNN_lofar.mat',
           {str(epochs) + '_train_acc_CNN_lofar': np.array(train_acc)})
io.savemat('CNN_lofar_train_acc_loss/' + str(epochs) + 'train_loss_CNN_lofar.mat',
           {str(epochs) + '_train_loss_CNN_lofar': np.array(train_loss)})

# save the real train data
if not os.path.exists('CNN_lofar_real_train_acc_loss'):
    os.mkdir('CNN_lofar_real_train_acc_loss')
io.savemat('CNN_lofar_real_train_acc_loss/' + str(epochs) + 'real_train_acc_CNN_lofar.mat',
           {str(epochs) + '_real_train_acc_CNN_lofar': np.array(real_train_acc)})
io.savemat('CNN_lofar_real_train_acc_loss/' + str(epochs) + 'real_train_loss_CNN_lofar.mat',
           {str(epochs) + '_real_train_loss_CNN_lofar': np.array(real_train_loss)})
# load model
# model = load_model('cnn_lstm_model.h5')

cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
print('test cost: ', cost, 'test accuracy: ', accuracy)
test_acc = [accuracy]
# test_acc = []
# test_acc.append(accuracy)
# save test results
if not os.path.exists('CNN_lofar_test_results'):
    os.mkdir('CNN_lofar_test_results')
io.savemat('CNN_lofar_test_results/' + str(epochs) + '.mat', {str(epochs): np.array(test_acc)})
m = io.loadmat('CNN_lofar_test_results/' + str(epochs) + '.mat')
print(m)

