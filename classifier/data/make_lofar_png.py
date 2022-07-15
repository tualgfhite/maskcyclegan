#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:42:16 2020

@author:
"""
import librosa
import matplotlib as mpl
from tqdm import tqdm

mpl.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
# from scipy.fftpack import fft
# import h5py as h5
import scipy.io as scio
# import matplotlib.pyplot as plt
# -*-coding:utf8-*-
import numpy as np  # 用于对 ndarray 进行操作，包括但不限于生成、组合、删除
from PIL import Image
from scipy import signal  # 用于对信号进行短时傅里叶操作

b = signal.firwin2(512, [0, 10*2/5000, 80*2/5000, 100*2/5000, 800*2/5000, 850*2/5000, 1200*2/5000, 2500*2/5000],
                   gain=[0, 0, 0, 1, 1, 0, 0, 0])  # FIR 滤波 100~800Hz
# b, a = signal.butter(3, [0.008, 0.24], btype='bandpass')  # IIR 滤波
# # b = signal.firwin(128, 0.5, window='hamming')
w, h = signal.freqz(b)

""" 1.定义 get_path() 函数，获取data_raw里面的所有mat文件的地址 """
"""  --------------------------  这里换下路径  --------------------------  """

absolute_path = '../../data/'
save_path1 = 'lofar_png/train'
save_path2 = 'lofar_png/test'

path1 = absolute_path + 'time_expanded/data_raw_trsec'
path2 = absolute_path + 'time_expanded/data_raw_tesec'

filter = [".mat"]  # 设置过滤后的文件类型 当然可以设置多个类型，此处是要选出 .mat 文件进行处理

apath = []
count_file = -1
iteration = 0
fs = 5000
# data0 = np.zeros(16384, dtype=complex)



def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


def createpng(path,save_path):
    for main_dir, sub_dir, file_name_list in os.walk(path):
        for filename in tqdm(file_name_list):
            apath = os.path.join(main_dir, filename)  # os.path.join() => 合并成一个完整路径
            ext = os.path.splitext(apath)[1]  # 获取文件后缀，以便之后筛选出 .mat 文件
            if ext == '.mat':
                mat = scio.loadmat(apath)
                d = mat['DataOut']
                # data_in_a_mat = np.transpose(d, (1, 0))
                data_in_a_mat = d.reshape(16384, )
                # data_in_a_mat[0:1024] = np.nanmin(abs(data_in_a_mat))  # 去掉头部冲击
                data_in_a_mat = data_in_a_mat / np.nanmax(abs(data_in_a_mat))  # 归一化

                # 并将它们 reshape 为一维数据方便做【短时傅里叶变换】

                data_in_a_mat = signal.filtfilt(b, 1, data_in_a_mat)
                f, t, Zxx = signal.stft(data_in_a_mat, fs, nperseg=256)
                lofar = np.abs(Zxx)[1:129, 1:129].reshape(128, 128)
                # plt.pcolormesh(t, f, np.abs(Zxx))
                lofar = lofar / lofar.max()
                lofar_img = MatrixToImage(lofar)
                lofar_img.save(os.path.join(save_path, main_dir.split('/')[-1],filename.split(".")[0] + '.png'))
            elif ext == '.wav':
                mat, _ = librosa.load(apath, sr=5000, mono=True)
                assert len(mat) == 16384
                data_in_a_mat = mat.reshape(16384, )
                # data_in_a_mat[0:1024] = np.nanmin(abs(data_in_a_mat))  # 去掉头部冲击
                data_in_a_mat = data_in_a_mat / np.nanmax(abs(data_in_a_mat))  # 归一化

                # 并将它们 reshape 为一维数据方便做【短时傅里叶变换】
                data_in_a_mat = signal.filtfilt(b, 1, data_in_a_mat)
                f, t, Zxx = signal.stft(data_in_a_mat, fs, nperseg=256)
                lofar = np.abs(Zxx)[1:129, 1:129].reshape(128, 128)
                # plt.pcolormesh(t, f, np.abs(Zxx))
                lofar = lofar / lofar.max()
                lofar_img = MatrixToImage(lofar)
                lofar_img.save(os.path.join(save_path, main_dir.split('/')[-1],filename.split(".")[0] + '.png'))
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    b = signal.firwin2(512, [0, 10 * 2 / 5000, 80 * 2 / 5000, 100 * 2 / 5000, 800 * 2 / 5000, 850 * 2 / 5000,
                             1200 * 2 / 5000, 2500 * 2 / 5000],
                       gain=[0, 0, 0, 1, 1, 0, 0, 0])  # FIR 滤波 100~800Hz
    # b, a = signal.butter(3, [0.008, 0.24], btype='bandpass')  # IIR 滤波
    # # b = signal.firwin(128, 0.5, window='hamming')
    w, h = signal.freqz(b)
    apath = []
    count_file = -1
    iteration = 0
    fs = 5000
    createpng(path1,save_path1)
    createpng(path2,save_path2)