#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:30:27 2019

@author: hnguyen
"""
import torch
import alphabets
from layers import LSTM, BidirectionalLSTM
import torch.nn as nn
import torch.nn.functional as F

# about data and net
alphabet = alphabets.alphabet
keep_ratio = False  # whether to keep ratio for image resize
manualSeed = 1234  # reproduce experiemnt
random_sample = True  # whether to sample the dataset with random sampler
imgH = 64  # the height of the input image to network
imgW = 128  # the width of the input image to network
nh = 512  # size of the lstm hidden state
nc = 3  # greyscale or RGB
dealwith_lossnan = True  # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True  # enables cuda
multi_gpu = False  # whether to use multi gpu
ngpu = 1  # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0  # number of data loading workers

# training process
displayInterval = 100  # interval to be print the train loss
valInterval = 1000  # interval to val the model loss and accuracy
saveInterval = 1000  # interval to save model
n_val_disp = 10  # number of samples to display when val the model

# finetune
nepoch = 1000  # number of epochs to train for
batchSize = 64  # input batch size
lr = 1e-4  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
adam = False  # whether to use adam (default is rmsprop)
adadelta = False  # whether to use adadelta (default is rmsprop)

label = 'classify'
classify_lambda = 0.01

class OCR(nn.Module):  # backup

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(OCR, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 1, 3, 1, 3, 1, 3, 1, (2, 3), (1, 3), (1, 3), (1, 3), (1, 3)]
        ps = [(0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        ss = [1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1]
        nm = [24, 18, 32, 24, 40, 32, 54, 40, 64, 128, 128, 128, 128]

        cnn = nn.Sequential()

        def convRelu(cnn, i, batchNormalization=True):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(cnn, 0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(cnn, 1, True)
        convRelu(cnn, 2, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(cnn, 3, True)
        convRelu(cnn, 4, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 1), (2, 1)))
        convRelu(cnn, 5, True)
        convRelu(cnn, 6, True)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 1), (2, 1)))
        convRelu(cnn, 7, True)
        convRelu(cnn, 8, True)
        convRelu(cnn, 9, True)

        cnn1 = nn.Sequential()
        convRelu(cnn1, 10, True)
        cnn2 = nn.Sequential()
        convRelu(cnn2, 11, True)
        cnn3 = nn.Sequential()
        convRelu(cnn3, 12, True)

        self.cnn = cnn
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.cnn3 = cnn3

        self.embedding   = nn.Conv2d(512,   192, 1, 1, 0)
        self.embedding_1 = nn.Conv2d(192, nclass, 1, 1, 0)

        self.embedding_class = nn.Conv2d(192, 1, 1, 1, 0)

    def forward(self, input):
        # conv features
        conv0 = self.cnn(input)
        conv1 = self.cnn1(conv0)
        conv2 = self.cnn2(conv1)
        conv3 = self.cnn3(conv2)

        #        print(conv0.size(),conv1.size(),conv2.size(),conv3.size())

        conv = torch.cat([conv0, conv1, conv2, conv3], dim=1)
        #        print(conv0.size(),conv1.size(),conv2.size(),conv3.size(),conv.size())
        b, c, h, w = conv.size()
        #        print(conv.size())
        conv = conv.permute(0, 3, 1, 2)  # [b, w, c, h]
        conv = conv.reshape(b, w, c * h, 1)
        conv = conv.permute(0, 2, 1, 3)
        #        print(conv.size())

        # 1st line
        conv = F.relu(self.embedding(conv))
        output = self.embedding_1(conv)
        output = output.squeeze(3)

        # Note: We need to do reshape here, then log_softmax, then reshape again due to compatibility with ONNX.
        # ONNX only supports 2D input with dim=1 for softmax.
        # Convert from (b, h, T) to (T, b, h)
        output = output.permute(2, 0, 1)
        T, b, h = output.size()
        output = output.reshape(T * b, h)
        output = F.log_softmax(output, dim=1)
        output = output.view(T, b, -1)
        # Convert back to (b, h, T)
        output = output.permute(1, 2, 0)

        class_output = self.embedding_class(conv)
        # print(class_output)
        # print(class_output.size())
        # class_output = class_output.squeeze(0)
        class_output = class_output.squeeze(1)
        class_output = class_output.squeeze(2)
        class_output = torch.max(class_output, 1)

        # print(class_output.size())
        # print(class_output)

        return output, class_output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # replace all nan/inf in gradients to zero
