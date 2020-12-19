import os
import logging
import numpy as np
import time

import torch
from torch.nn import init
import torch.nn.functional as F

import torch.utils.data as data
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
            
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def norm(x):

    n = np.linalg.norm(x)
    return x / n

def val(loader, args, t_model, s_model, logger, epoch):

    s_model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in loader:

        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            _, output = s_model(x, is_feat=True)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))
        loss_record.update(loss.item(), x.size(0))

    run_time = time.time() - start
    if logger is not None:
        logger.add_scalar('val/cls_loss', loss_record.avg, epoch+1)
        logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

        info = 'student_test_Epoch:{:03d}\t run_time:{:.2f}\t cls_acc:{:.2f}\n'.format(
                epoch+1, run_time, acc_record.avg)
        print(info)
    return acc_record.avg

def cal_center(loader, args, model):

    model.eval()
    feat = []
    label = []
    for x, target in loader:

        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            batch_feat, output = model(x, is_feat=True)
        feat.append(batch_feat[-1])
        label.append(target)

    feat = torch.cat(feat, dim=0).cpu().numpy()
    label = torch.cat(label, dim=0).cpu().numpy()

    center = []
    for i in range(max(label)+1):
        index = np.where(label==i)[0]
        center.append(np.mean(feat[index], axis=0))
    center = np.vstack(center)
    center = torch.from_numpy(center).cuda()

    return center


