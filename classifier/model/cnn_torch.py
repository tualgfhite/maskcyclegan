# -*- coding:utf-8 -*-
"""
project: MaskCycleGANproject_time
file: cnn_torch.py
author: Jiang Xiangyu
create date: 7/7/2022 11:59 PM
description: None
"""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_size,kernel_nums,expand=True,num_class=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, kernel_nums, kernel_size=6),
            nn.BatchNorm2d(kernel_nums),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(kernel_nums, kernel_nums*2, kernel_size=6),
            nn.BatchNorm2d(kernel_nums*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(kernel_nums*2, kernel_nums*2, kernel_size=6),
            nn.BatchNorm2d(kernel_nums*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(kernel_nums*2, kernel_nums*2, kernel_size=6),
            nn.BatchNorm2d(kernel_nums*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Sequential(
            nn.Flatten(1,-1),
            # nn.Dropout(0.3)
        )
        self.dense = nn.Sequential(
            nn.Linear(30976 , 1024),
            nn.Linear(1024, num_class)
        )
    def forward(self, x):
        # x=x.unsqueeze(2).repeat(1,1,self.cols).unsqueeze(1).float()
        #print('x.repeat:'+x.shape())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    x=torch.randn([256,1,128,128])
    model = CNN(1,128,num_class=4)
    print(model(x).size())