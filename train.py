# coding: utf-8

# 调库一把梭
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader

from utils import get_dataset, evlation

# 指定gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2, 3, 4]


def update_optim(params, lr, momentum, weight_decay):
    optimizer = optim.SGD(params,
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    return optimizer


if __name__ == '__main__':
    # 超参设置
    LR = 1e-2
    batch_size = 128
    num_epoch = 200
    weight_decay = 1e-4
    momentum = 0.9

    model_name = 'resnet50'
    dataset_name = 'cifar100'
    # 数据集载入
    trainloader, testloader = get_dataset(batch_size * len(device_ids),
                                          dataset_dir='./')

    # 定义模型
    net = resnet50()
    net.cuda()
    # 并行方式
    # net = nn.DataParallel(net, device_ids=device_ids)
    """
    方式一，推荐
    save网络参数
    torch.save(net.state_dict(), fname)
    载入模型
    net.load_state_dict(torch.load(fname))

    方式二
    save网络结构和参数
    torch.save(net, fname)
    net = torch.load(fname)
    """
    
    """
    fine-tuning 模型
    net = resnet50(pretrained=True)
    net.cuda()
    # 修改最后一层全连接层，xxx可保持不变. num_classes输出需对应分类数目，比如cifar10为10，cifar100为100
    net.fc = Linear(xxx, num_classes)
    """

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(net.parameters(),
                          lr=LR,
                          momentum=momentum,
                          weight_decay=weight_decay)

    for epoch in range(num_epoch):
        s1 = time.time()
        net.train()

        # if epoch >= 100:
        #     optimizer = update_optim(net.parameters(), 0.01, 0.9, 5e-4)
        # if epoch >= 150:
        #     optimizer = update_optim(net.parameters(), 0.001, 0.9, 5e-4)

        for i, data in enumerate(trainloader):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        e1 = time.time()
        print('[%d, %5d] loss: %.4f time: %.4f' %
              (epoch + 1, (i + 1) * batch_size, loss.item(), e1 - s1))


        evlation(net, testloader, device)

        print('Saving model......')
        fname = '%s/net_%03d.pth' % (model_name + '-' + dataset_name,
                                        epoch + 1)
        torch.save(net.state_dict(), fname)
