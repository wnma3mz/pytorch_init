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

def get_dataset(batch_size, dataset_dir='./'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR100(root=dataset_dir,
                        train=True,
                        download=False,
                        transform=transform_train)

    testset = CIFAR100(root=dataset_dir,
                       train=False,
                       download=False,
                       transform=transform_test)

    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    return trainloader, testloader



def evlation(net, testloader, device):
    # 评估
    net.eval()
    time_start = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            # images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 预测正确数

    print('Acc: {}'.format(100 * correct / total))
    time_end = time.time()
    print('Time cost:', time_end - time_start, "s")