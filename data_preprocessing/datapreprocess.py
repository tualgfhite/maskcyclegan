# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pickle

import numpy as np
from scipy import signal
from scipy.io import loadmat
from tqdm import tqdm


def normalize_lofar(matspath):
    mat_files = glob.glob(os.path.join(
        matspath, '**', '*.mat'), recursive=True)  # source_path
    lofar_list = list()
    for matpath in tqdm(mat_files, desc='Preprocess mat to lofar'):
        spec = mat_to_lofar(matpath)
        if spec.shape[-1] >= 128:  # training sample consists of 64 randomly cropped frames
            lofar_list.append(spec)
    lofar_concatenated = np.concatenate(lofar_list, axis=1)
    lofar_mean = np.mean(lofar_concatenated, axis=1, keepdims=True)
    lofar_std = np.std(lofar_concatenated, axis=1, keepdims=True) + 1e-9
    lofar_normalized = list()
    for lofar in lofar_list:
        assert lofar.shape[
                   -1] >= 128, f"lofar spectogram length must be greater than 128 frames, but was {lofar.shape[-1]}"
        app = (lofar - lofar_mean) / lofar_std
        lofar_normalized.append(app)
    return lofar_normalized, lofar_mean, lofar_std


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_dataset(data_path, boat_id, cache_folder='./cache/'):
    """Preprocesses dataset of .mat files by converting to lofar-spectrograms.
    Args:
        data_path (str): Directory containing .mat files of the boat.
        boat_id (str): ID of the boat.
        cache_folder (str, optional): Directory to hold preprocessed data. Defaults to './cache/'.
    """
    print(f"Preprocessing data for boat: {boat_id}.")
    lofar_normalized, lofar_mean, lofar_std = normalize_lofar(data_path)
    if not os.path.exists(os.path.join(cache_folder, str(boat_id))):
        os.makedirs(os.path.join(cache_folder, str(boat_id)))
    np.savez(os.path.join(cache_folder, str(boat_id), f"{str(boat_id)}_norm_stat.npz"),
             mean=lofar_mean,
             std=lofar_std)
    save_pickle(variable=lofar_normalized,
                fileName=os.path.join(cache_folder, str(boat_id), f"{str(boat_id)}_normalized.pickle"))
    print(f"Preprocessed and saved data for boat: {str(boat_id)}.")


def mat_to_lofar(matfile):
    b = signal.firwin2(512, [0, 10 * 2 / 5000, 80 * 2 / 5000, 100 * 2 / 5000, 800 * 2 / 5000, 850 * 2 / 5000,
                             1200 * 2 / 5000, 2500 * 2 / 5000],
                       gain=[0, 0, 0, 1, 1, 0, 0, 0])  # FIR 滤波 100~800Hz
    fs = 5000
    mat = loadmat(matfile)
    # print(list(mat.keys()))
    keylist = list(mat.keys())
    data1 = mat[keylist[3]]
    data_in_a_mat = data1.reshape(16384, )
    # data_in_a_mat[0:1024] = np.nanmin(abs(data_in_a_mat))  # 去掉头部冲击
    data_in_a_mat = data_in_a_mat / np.nanmax(abs(data_in_a_mat))  # 归一化
    # 并将它们 reshape 为一维数据方便做【短时傅里叶变换】
    data_in_a_mat = signal.filtfilt(b, 1, data_in_a_mat)
    f, t, Zxx = signal.stft(data_in_a_mat, fs, nperseg=256)
    lofar = np.abs(Zxx)[1:129, 1:129].reshape(128, 128)
    # plt.pcolormesh(t, f, np.abs(Zxx))
    lofar = lofar / lofar.max()
    return lofar


# --data_directory '../data/time_expand/data_raw_trsec/1' --preprocessed_data_directory 'data_preprocessed/time_training' --day_ids ['4', '6']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_directory', type=str, default='../data/data_raw',
                        help='Directory holding dataset.')
    parser.add_argument('--preprocessed_data_directory', type=str, default='data_preprocessed/mini',
                        help='Directory holding preprocessed dataset.')
    parser.add_argument('--day_ids', nargs='+', type=str, default=[1, 2, 3,4],
                        help='Source day id')
    args = parser.parse_args()
    for day_id in args.day_ids:
        data_path = os.path.join(args.data_directory, str(day_id))
        preprocess_dataset(data_path=data_path, boat_id=day_id,
                           cache_folder=args.preprocessed_data_directory)
