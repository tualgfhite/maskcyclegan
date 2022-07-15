# -*- coding:utf-8 -*-
"""
project: MaskCycleGANproject_time
file: classifierdataset.py
author: Jiang Xiangyu
create date: 7/4/2022 9:09 PM
description: None
"""
import os
import pickle
import warnings

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
import numpy as np

def read_data( datadir, split="train"):
    datapath = os.path.join(datadir, split, f'{split}_signal.pickle')
    with open(datapath, 'rb') as datafile:
        path_signals = pickle.load(datafile)
    labelpath = os.path.join(datadir, split, f'y_{split}.pickle')
    with open(labelpath, 'rb') as labelfile:
        labels = pickle.load(labelfile)

    # Return
    return  path_signals, labels.squeeze()

class Mydataset(Dataset):
    def __init__(self, dataset,label,standard=False,onehot=False):
        self.audio,self.target = dataset,label
        if standard:
            self.audio = self._standardization(self.audio)
        if onehot:

            self.target = self._one_hot(self.target)
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])

    def __getitem__(self, index):
        return self.audio[index], self.target[index]
    def __len__(self):
        return len(self.audio)
    def _standardization(self,dataset):
        mean = np.mean(dataset, axis=0)
        std = np.std(dataset, axis=0)
        dataset = (dataset - mean[None, :, :]) / std[None, :, :]
        return dataset

    def _one_hot(self,labels):
        """ One-hot encoding """
        mask = np.unique(labels)
        counts = []
        for v in mask:
            counts.append(np.sum(labels == v))
        counts=np.sum(counts)/counts
        expansion = np.diag(counts)
        y = expansion[:, labels - 1].T
        return y

if __name__ == "__main__":
    X_train,Y_train=read_data(datadir='pickle_time', split='train')
    X_test,Y_test=read_data(datadir='pickle_time', split='test')
    traindataset=Mydataset(X_train.reshape([-1,1,128,128]),Y_train,onehot=True)
    testdataset = Mydataset(X_test.reshape([-1,1,128,128]), Y_test)
