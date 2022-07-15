# 2020.10.13
# from lofar-pic to dataset
import pickle

import librosa
import matplotlib as mpl
from scipy import signal
from scipy.io import loadmat
from tqdm import tqdm

mpl.use('TkAgg')
import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)

import numpy as np
import os


def get_lofar(wav):
    b = signal.firwin2(512, [0, 10 * 2 / 5000, 80 * 2 / 5000, 100 * 2 / 5000, 800 * 2 / 5000, 850 * 2 / 5000,
                             1200 * 2 / 5000, 2500 * 2 / 5000],
                       gain=[0, 0, 0, 1, 1, 0, 0, 0])
    fs = 5000
    # data_in_a_mat[0:1024] = np.nanmin(abs(data_in_a_mat))  # 去掉头部冲击
    assert len(wav) == 16384
    data_in_a_mat = wav.reshape(16384, )
    data_in_a_mat = data_in_a_mat / np.nanmax(abs(data_in_a_mat))  # 归一化
    # 并将它们 reshape 为一维数据方便做【短时傅里叶变换】
    data_in_a_mat = signal.filtfilt(b, 1, data_in_a_mat)
    f, t, Zxx = signal.stft(data_in_a_mat, fs, nperseg=256)
    lofar = np.abs(Zxx)[1:129, 1:129].reshape(128, 128)
    # plt.pcolormesh(t, f, np.abs(Zxx))
    lofar = lofar / lofar.max()
    return lofar


def readlofar(path):
    labels = np.zeros((1, 1))  # 所有的标签
    # ch_labels = np.zeros((1, 1))
    data_row_lof = np.zeros((1, 128, 128))
    # lab = os.path.split(path)[-1]  # 等于path最后一个数字，如path=“data2type/1”时，lab=1
    # lab = 4  # 等于path最后一个数字，如path=“data2type/1”时，lab=1
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in tqdm(file_name_list):
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            if apath.split('.')[-1] == 'mat':
                mat = loadmat(apath)
                data1 = mat[list(mat.keys())[3]]
                wavarray = data1.reshape(16384, )
            elif apath.split('.')[-1] == 'wav':
                wavarray, _ = librosa.load(apath, sr=5000, mono=True)
            else:
                continue
            labels = np.vstack((labels, os.path.split(maindir)[-1]))
            data_in_a_mat = get_lofar(wavarray).reshape([1, 128, 128])
            data_row_lof = np.vstack((data_row_lof, data_in_a_mat))
    return data_row_lof, labels


def topickle(time_or_mini=True):
    if time_or_mini:
        datapath1 = "../../data/time/data_raw_trsec"
        datapath2 = "../../data/time/data_raw_tesec"
        savepath = "pickle_time"
        labels_tr, data_row_lof_tr = readlofar(datapath1)
        labels_te, data_row_lof_te = readlofar(datapath2)
        # #######储存训练信道数据集#####################上
        labels_tr = np.delete(labels_tr, 0, axis=0)  # axis = 0 表示删除行，=1 表示删除列，括号中间的‘0’表示删除维度的序数
        labels_te = np.delete(labels_te, 0, axis=0)  # axis = 0 表示删除行，=1 表示删除列，括号中间的‘0’表示删除维度的序数
        data_row_lof_tr = np.delete(data_row_lof_tr, 0, axis=0)
        data_row_lof_te = np.delete(data_row_lof_te, 0, axis=0)
        # 数据和标签一起打乱顺序
        permutation = np.random.permutation(labels_tr.shape[0])
        permutation2 = np.random.permutation(labels_te.shape[0])
        shuffled_labels_tr = labels_tr[permutation]
        shuffled_labels_te = labels_te[permutation2]
        shuffled_dataset_lof_tr = data_row_lof_tr[permutation, :]
        shuffled_dataset_lof_te = data_row_lof_te[permutation2, :]
        shuffled_labels_tr = shuffled_labels_tr.astype(np.int16)
        shuffled_labels_te = shuffled_labels_te.astype(np.int16)

        output1 = open(f"{savepath}/train/train_signal.pickle", 'wb')
        pickle.dump(shuffled_dataset_lof_tr, output1)
        output2 = open(f"{savepath}/test/test_signal.pickle", 'wb')
        pickle.dump(shuffled_dataset_lof_te, output2)
        output3 = open(f"{savepath}/train/y_train.pickle", 'wb')
        pickle.dump(shuffled_labels_tr, output3)
        output4 = open(f"{savepath}/test/y_test.pickle", 'wb')
        pickle.dump(shuffled_labels_te, output4)
    else:
        savepath = "pickle_mini"
        data_row_lof, labels = readlofar('../../data/data_raw')
        labels = np.delete(labels, 0, axis=0)
        data_row_lof = np.delete(data_row_lof, 0, axis=0)
        permutation = np.random.permutation(labels.shape[0])
        shuffled_labels = labels[permutation]
        shuffled_dataset_lof = data_row_lof[permutation, :, :]
        shuffled_labels = shuffled_labels.astype(np.int16)

        output1 = open(f"{savepath}/train/train_signal.pickle", 'wb')
        pickle.dump(shuffled_dataset_lof[:shuffled_dataset_lof.shape[0] // 2, :, :], output1)
        output2 = open(f"{savepath}/test/test_signal.pickle", 'wb')
        pickle.dump(shuffled_dataset_lof[shuffled_dataset_lof.shape[0] // 2:, :, :], output2)
        output3 = open(f"{savepath}/train/y_train.pickle", 'wb')
        pickle.dump(shuffled_labels[:shuffled_labels.shape[0] // 2], output3)
        output4 = open(f"{savepath}/test/y_test.pickle", 'wb')
        pickle.dump(shuffled_labels[shuffled_labels.shape[0] // 2:], output4)


if __name__ == '__main__':
    # with open("./pickle_mini/train/train_signal.pickle", 'rb')as f:
    #     data=pickle.load(f)
    # with open("./pickle_mini/test/test_signal.pickle", 'rb')as f:
    #     data2=pickle.load(f)
    topickle(False)
