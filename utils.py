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


def evaluate(test_loader,
             model,
             criterion,
             n_iter=-1,
             verbose=False,
             device='cuda'):
    # 标准评估方式。计算top1, top5
    n_iter = len(test_loader) if n_iter == -1 else n_iter
    modulo = 0 if verbose else -1
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_loader),
                             batch_time,
                             losses,
                             top1,
                             top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(test_loader):
            # early stop
            if i >= n_iter: break

            # cuda
            input_ = input_.cuda() if device == 'cuda' else input_
            target = target.cuda() if device == 'cuda' else target

            # compute output
            output = model(input_)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input_.size(0))
            top1.update(acc1[0], input_.size(0))
            top5.update(acc5[0], input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == modulo:
                progress.print(i)

        return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter(object):
    """
    Pretty and compact metric printer.
    """
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
