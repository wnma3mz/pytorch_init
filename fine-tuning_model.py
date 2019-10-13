# coding: utf-8

import os
import time
import math
import argparse
from operator import attrgetter
from bisect import bisect_left

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.optim as optim

import models
from data import load_data
from optim import CentroidSGD
from quantization import PQ
from utils.training import finetune_centroids, evaluate
from utils.watcher import ActivationWatcher
from utils.dynamic_sampling import dynamic_sampling
from utils.statistics import compute_size
from utils.utils import centroids_from_weights, weight_from_centroids

data_path = '/home/lujianghu/kill-the-bits/src/cifar10/'
data_path = '/home/lujianghu/kill-the-bits/src/'
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

batch_size = 128
n_workers = 20
init_epoch = 0
num_epoch = 100
LR = 0.001
momentum = 0.9
weight_decay = 5e-4
dataset_name = 'cifar100'
model_name = 'ResNet18'

if __name__ == '__main__':
    net = models.__dict__['resnet18'](pretrained=True).cuda()
    
    net.fc=nn.Linear(512,100,bias=True)
    # net = models.__dict__['resnet34'](num_classes=100)

    # net.load_state_dict(torch.load('ResNet50-cifar10/net_050.pth'))
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)
    cudnn.benchmark = True

    train_loader, test_loader = load_data(data_path=data_path, batch_size=batch_size, nb_workers=n_workers)

    # data loading code

    for epoch in range(init_epoch, num_epoch):

        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        s1 = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        e1 = time.time()

        print('[%d, %5d] loss: %.4f time: %.4f' % (epoch + 1,
                                        (i + 1) * batch_size, loss.item(), e1 - s1))
        
        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).cpu().sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            
            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            fname = '%s/net_%03d.pth' % (model_name + '-' + dataset_name,
                                         epoch + 1)
            torch.save(net.state_dict(), fname)
    print("Training Finished, TotalEPOCH=%d" % epoch)
