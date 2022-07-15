# -*- coding:utf-8 -*-
"""
project: MaskCycleGANproject_time
file: Resnet_tf.py
author: Jiang Xiangyu
create date: 7/3/2022 12:05 PM
description: None
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, \
    Dense, add
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.keras import losses
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam


class BasicBlock(Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock,self).__init__()

        self.conv1=Conv2D(filter_num,kernel_size=(3,3),strides=stride,padding='same')
        self.bn1=BatchNormalization()
        self.relu1=Activation('relu')

        self.cov2=Conv2D(filter_num,kernel_size=(3,3),strides=1,padding='same')
        self.bn2=BatchNormalization()
        self.relu2=Activation('relu')

        if stride != 1:
            self.downsample=Sequential()
            self.downsample.add(Conv2D(filter_num,(1,1),strides=stride,padding='same'))
        else:
            self.downsample=lambda x:x
    def call(self, inputs, training=None):
        #input[N.H,W,C]
        out=self.conv1(inputs)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.cov2(out)
        out=self.bn2(out)
        identity=self.downsample(inputs)
        output=add([out,identity])
        output=tf.nn.relu(output)
        return output

class Resnet(Model):
    def __init__(self,layer_dim,num_classes=3):
        super(Resnet,self).__init__()
        #input[N.H,W,C]
        self.stem=Sequential([Conv2D(64,kernel_size=(3,3),strides=1,padding='same'),
                             BatchNormalization(),
                             Activation('relu'),
                             MaxPool2D(pool_size=(2,2),strides=1,padding='same')])
        self.layer1=self.build_resblock(64,layer_dim[0])
        self.layer2 = self.build_resblock(128, layer_dim[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dim[2],stride=2)
        self.layer4 = self.build_resblock(512, layer_dim[3],stride=2)
        self.avgpool = GlobalAveragePooling2D()
        self.fc1 = Dense(1024,activation='relu')
        self.fc2=Dense(num_classes,activation='softmax')

    def build_resblock(self,filter_num,blocks,stride=1):
        res_block=Sequential()
        res_block.add(
            BasicBlock(filter_num,stride)
        )
        for _ in range(1,blocks):
            res_block.add(BasicBlock(filter_num,stride=1))
        return res_block
    def call(self,inputs,training=None):
        x=self.stem(inputs)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.avgpool(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return x
if __name__=='__main__':
    resnet18=Resnet([2,2,2,2],100)
    resnet34 = Resnet([3, 4, 6, 3], 100)
    x=np.random.random([64,128,128,1])
    print(resnet18(x).shape)