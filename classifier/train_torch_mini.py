# -*- coding:utf-8 -*-
"""
project: MaskCycleGANproject_time
file: train_torch.py
author: Jiang Xiangyu
create date: 7/4/2022 9:01 PM
description: None
"""
# -*- coding:utf-8 -*-
import argparse
import pickle
import warnings

import torch
from matplotlib import pyplot as plt
from torch import nn

from data.classifierdataset import Mydataset
from model.Resnet_torch2 import Resnet, BasicBlock
from model.cnn_torch import CNN

warnings.filterwarnings('ignore')
import os
import numpy as np


def read_data(datadir, split="train"):
    datapath = os.path.join(datadir, split, f'{split}_signal.pickle')
    with open(datapath, 'rb') as datafile:
        path_signals = pickle.load(datafile)
    labelpath = os.path.join(datadir, split, f'y_{split}.pickle')
    with open(labelpath, 'rb') as labelfile:
        labels = pickle.load(labelfile)
    return path_signals, labels.squeeze()


def update_lr(optimizer, lr):
    lr /= 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def readdata2(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def smooth(data, ss=1):
    if ss > 1:
        y = np.ones(ss) / float(ss)
        data = np.hstack((data, data[1 - ss:]))
        data = np.convolve(data, y, "valid")
    return data


def plot(accs, pngname, smoothstep=5):
    '''
    先画各类，再画总acc
    '''
    for i in range(accs.shape[1] - 1):
        plt.plot(smooth(accs[:, i], smoothstep), label=f'cat{i + 1}acc')
    plt.plot(smooth(accs[:, -1], smoothstep), label='total_acc')
    plt.ylabel("acc")
    plt.xlabel("epochs")
    plt.title(f"{pngname}")
    plt.legend()
    plt.savefig(f"../results/{pngname}.png", bbox_inches='tight')


def standardization(datasetA, datasetB=None):
    if datasetB is None:
        mean = np.mean(datasetA, axis=0)
        std = np.std(datasetA, axis=0)
        dataset = (datasetA - mean[None, :, :]) / std[None, :, :]
        return dataset
    else:
        # dataall = np.vstack((datasetA, datasetB))
        mean = np.mean(datasetA, axis=0)
        std = np.std(datasetA, axis=0)
        datasetA = (datasetA - mean[None, :, :]) / std[None, :, :]
        datasetB = (datasetB - mean[None, :, :]) / std[None, :, :]
        return datasetA, datasetB


if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser(description='classifier')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='classifier learn rate')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for dataloader')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--expand', action='store_true',
                        help='train with expanded data?')
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='N',
                        help='normalization')
    parser.add_argument('--model_choice', type=str, default='CNN',
                        help='which model?(resnet34/resnet18/CNN)')
    parser.add_argument('--num_class', type=int, default=3,
                        help='how many classes?')
    parser.add_argument('--datadir', type=str, default="data/pickle_mini",
                        help='original data directory')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(args)
    # 加载数据
    X_train, Y_train = read_data(datadir=args.datadir, split='train')
    X_test, Y_test = read_data(datadir=args.datadir, split='test')
    if args.expand:
        for i, j in [(1, 4), (2, 4), (3, 4)]:
            path = os.path.join(f'../results/mask_cyclegan_mini_{i}_{j}/converted_audio/converted_{i}_to_{j}.pickle')
            X_expand = readdata2(path)
            Y_expand = np.full([X_expand.shape[0]], fill_value=j)
            X_train = np.vstack((X_train, X_expand))
            Y_train = np.hstack((Y_train, Y_expand))
    # X_train,X_test=standardization(X_train,X_test)
    traindataset = Mydataset(X_train.reshape([-1, 1, 128, 128]), Y_train, onehot=True)
    testdataset = Mydataset(X_test.reshape([-1, 1, 128, 128]), Y_test)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=args.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=4)
    # 模型配置
    if args.model_choice == 'resnet34':
        model = Resnet(BasicBlock, [3, 4, 6, 3], args.num_class)
    elif args.model_choice == 'resnet18':
        model = Resnet(BasicBlock, [2, 2, 2, 2], args.num_class)
    else:
        model = CNN(1, 128, num_class=args.num_class)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(train_loader)
    curr_lr = args.lr
    # plot acc
    accs = np.zeros([1, args.num_class + 1])
    single = np.zeros([args.num_class])
    for v in np.unique(Y_test - 1):
        single[v] = np.sum(Y_test - 1 == v)
    for epoch in range(args.num_epochs):
        all_loss = 0
        model.train()
        for i, (audio, labels) in enumerate(train_loader):
            audio = audio.type(torch.FloatTensor).to(args.device)
            labels = labels.to(args.device)
            outputs = model(audio)
            loss = criterion(outputs, labels) / args.batch_size
            all_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 学习率衰减
        update_lr(optimizer, curr_lr)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            each = np.zeros([args.num_class])  # 统计单类正确数
            for i, (audio, labels) in enumerate(test_loader):
                audio = audio.type(torch.FloatTensor).to(args.device)
                labels = (labels - 1).type(torch.int32)
                outputs = model(audio)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                predicted = predicted.cpu()
                for i in range(labels.size(0)):
                    if labels[i] == predicted[i]:
                        correct += 1
                        each[labels[i]] += 1
            print('Epoch [{}/{}],  Loss: {:.4f},test acc: {:.2f} %'.format(epoch + 1, args.num_epochs, all_loss,
                                                                           100 * correct / total))
            accs = np.vstack((accs, np.hstack((each / single, correct / total)).reshape(1, -1)))
    # 保存acc曲线，保存模型参数
    plot(accs, f'acc_{"expand" if not args.expand else "ori"}')
    torch.save(model.state_dict(),
               f'../results/mini_{args.model_choice}_{"expand" if args.expand else "ori"}_{np.average(accs[-10:, -1]):.2f}.pth')
