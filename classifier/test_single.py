# HAR classification
# Author: Burak Himmetoglu
# 8/15/2017
import matplotlib as mpl
from tensorflow.keras.optimizers import Adam

mpl.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from scipy import io
np.random.seed(1337)  # for reproducibility


def read_data(data_path, split="train"):
    """ Read data """

    # Fixed params
    # n_class = 4
    n_steps = 16384

    # Paths
    path_ = os.path.join(data_path, split)
    path_signals = os.path.join(path_)

    # Read labels and one-hot encode
    label_path = os.path.join(path_, "y_" + split)  # + ".txt")
    labels = pd.read_csv(label_path, header=None)

    # Read time_expand-series data
    n_channels = 1

    # Initiate array
    x = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    dat_ = pd.read_csv(os.path.join(path_signals, split + '_signal'), delim_whitespace=True, header=None)
    x[:, :, i_ch] = dat_
    # Return
    return x, labels[0].values


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


def get_batches(x, y, batch_size=100):
    """ Return a generator for batches """
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(x), batch_size):
        yield x[b:b + batch_size], y[b:b + batch_size]


# TIME_STEPS = 128     # same as the height of the image
# INPUT_SIZE = 128     # same as the width of the image
BATCH_SIZE = 256
BATCH_INDEX = 0
OUTPUT_SIZE = 4
CELL_SIZE = 50
LR = 0.001
TRAIN_VALIDATION_RATIO = 0.9999
epochs = 2000  # 这个决定训练量大小，现在是训练10000*20=20万个数据，假如总数据量是4万个，那么将把所有数据训练5次



def get_single_acc(single,type):#测试的船只种类;数据集种类
    # ## 注意路径中不能包含硬盘名字，如 C: 、 F: ，必须以硬盘名字之后的目标文件开头，且路径首末要有 “ / ” ####
    if type == "test":
        X_test, labels_test = read_data(data_path="lofar_data/", split="test")  # test
        X_test = X_test.reshape((-1, 16384))
        X_test_single = list()
        Y_test_single = list()

        # print(len(labels_test))
        for i in range(len(labels_test)):
            # print(labels_test[i])
            if labels_test[i] == single:
                Y_test_single.append(labels_test[i])
                X_test_single.append(X_test[i])
        Y_test_single = np.array(Y_test_single)  # axis = 0 表示删除行，=1 表示删除列，括号中间的‘0’表示删除维度的序数
        X_test_single = np.array((X_test_single))

        # One-hot encoding:
        Y_test_single = one_hot(Y_test_single)

        X_test_single = X_test_single.reshape((-1, 1, 128, 128))
    elif type =="train": # 训练集中第三类数据的识别准确率（包含原本的数据和生成样本）
        X_test, labels_test = read_data(data_path="lofar_data/", split="train")
        X_test = X_test.reshape((-1, 16384))
        X_test_single = list()
        Y_test_single = list()

        # print(len(labels_test))
        for i in range(len(labels_test)):
            # print(labels_test[i])
            if labels_test[i] == single:
                Y_test_single.append(labels_test[i])
                X_test_single.append(X_test[i])
        Y_test_single = np.array(Y_test_single)  # axis = 0 表示删除行，=1 表示删除列，括号中间的‘0’表示删除维度的序数
        X_test_single = np.array((X_test_single))

        # One-hot encoding:
        Y_test_single = one_hot(Y_test_single)

        X_test_single = X_test_single.reshape((-1, 1, 128, 128))
    elif type =="all_test": #测试所有测试集的准确率
        X_test, labels_test = read_data(data_path="lofar_data/", split="test")
        Y_test_single = one_hot(labels_test)
        X_test_single = X_test.reshape((-1, 1, 128, 128))


    # build  model
    model = Sequential()

    # build CNN model
    # Conv layer 1 output shape (4, 16384)
    model.add(Convolution2D(
        batch_input_shape=(None, 1, 128, 128),
        filters=4,
        kernel_size=5,
        strides=1,
        padding='same',  # Padding method
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

    # load model
    model = load_model('cnn_lofar_model.h5')

    cost, accuracy = model.evaluate(X_test_single, Y_test_single, batch_size=Y_test_single.shape[0], verbose=False)
    print('type:', single,'test cost: ', cost, 'test accuracy: ', accuracy)
    test_acc = [accuracy]
    # test_acc = []
    # test_acc.append(accuracy)
    # save test results
    #if not os.path.exists('CNN_lofar_test_results_4th'):
    #    os.mkdir('CNN_lofar_test_results_4th')
    #io.savemat('CNN_lofar_test_results_'+ str(single) +'th/' + str(epochs) + '.mat', {str(epochs): np.array(test_acc)})
    #m = io.loadmat('CNN_lofar_test_results_'+ str(single) +'th/' + str(epochs) + '.mat')

print("测试集结果")
get_single_acc(1,"all_test")#第一个参数随意写不影响

print("测试训练集中某类样本（包含原始数据）的识别效果")
get_single_acc(1,"train")
get_single_acc(2,"train")
get_single_acc(3,"train")
print("cnn_lofar_model")
for i in range(2):
    get_single_acc(i+2,'test')


'''
测试集结果
type: 1 test cost:  0.26162639260292053 test accuracy:  0.9001097679138184
测试训练集中某类样本（包含原始数据）的识别效果
type: 1 test cost:  0.2146279215812683 test accuracy:  0.9315263628959656
type: 2 test cost:  0.27608364820480347 test accuracy:  0.8931924700737
type: 3 test cost:  0.18302984535694122 test accuracy:  0.9306177496910095
cnn_lofar_model
type: 2 test cost:  0.39145320653915405 test accuracy:  0.8566243052482605
type: 3 test cost:  0.06291927397251129 test accuracy:  0.9666666388511658
'''


