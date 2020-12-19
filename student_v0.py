import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from models import model_dict
from zoo import Attention, Similarity
from dataset import CIFAR100
from utils import accuracy, val, AverageMeter, cal_center

items = ['acc', 'loss', \
        's_select_confidence', 's_select_margin', 's_select_entropy', \
        's_else_confidence', 's_else_margin', 's_else_entropy', \
        's_all_confidence', 's_all_margin', 's_all_entropy', \
        't_confidence', 't_margin', 't_entropy', \
        'center_dist']

parser = argparse.ArgumentParser(description='train student network.')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--k', type=int, default=48)
parser.add_argument('--b', type=int, default=32)
parser.add_argument('--w', type=float, default=1000)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--milestones', type=float, nargs='+', default=[150, 180, 210])

parser.add_argument('--teacher-path', type=str, default='./experiments/teacher_resnet32x4')
parser.add_argument('--teacher-ckpt', type=str, default='best')
parser.add_argument('--student-arch', type=str, default='resnet8x4')

parser.add_argument('--ce-weight', type=float, default=0.0)
parser.add_argument('--kd-weight', type=float, default=1.0)
parser.add_argument('--other-distill', type=str, choices=['AT', 'SP'], default=None)
parser.add_argument('--T', type=float, default=4.0)

parser.add_argument('--strategy', type=int, choices=[0,1,2,3], default=3)
# 0: random, 1: least confidence, 2: margin, 3: entropy

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

torch.backends.cudnn.benchmark = True

teacher_arch = '_'.join(args.teacher_path.split('/')[-1].split('_')[1:])
exp_name = '{}_student_{}_teacher_{}-{}_strategy{}_k{}_b{}_w{}_seed{}'.format(\
            __file__.split('.')[0].split('_')[-1], \
            args.student_arch, teacher_arch, args.teacher_ckpt, \
            args.strategy, \
            args.k, args.b, args.w, \
            args.seed)

exp_path = './experiments/{}'.format(exp_name)
os.makedirs(exp_path, exist_ok=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])

trainset = CIFAR100('./data', train=True, transform=transform_train)
valset = CIFAR100('./data', train=False, transform=transform_test)
num_classes = 100

train_loader = DataLoader(trainset, batch_size=args.batch_size, \
            shuffle=True, num_workers=3, pin_memory=True)
val_loader = DataLoader(valset, batch_size=args.batch_size, \
            shuffle=False, num_workers=3, pin_memory=True)

ckpt_path = osp.join('{}/ckpt/{}.pth'.format( \
                args.teacher_path, args.teacher_ckpt))
t_model = model_dict[teacher_arch](num_classes=num_classes).cuda()
state_dict = torch.load(ckpt_path)['state_dict']
t_model.load_state_dict(state_dict)
t_model.eval()

logger = SummaryWriter(osp.join(exp_path, 'events'))

s_model = model_dict[args.student_arch](num_classes=num_classes).cuda()
optimizer = optim.SGD(s_model.parameters(), lr=args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.milestones)

if args.other_distill is not None:
    if args.other_distill == 'AT':
        criterion = Attention()
        weight = 1000
    elif args.other_distill == 'SP':
        criterion = Similarity()
        weight = 3000

best_acc = 0
counter = torch.zeros(args.epoch, 50000).cuda()
epoch = 0

for epoch in range(args.epoch):

    record = {name:AverageMeter() for name in items}
    center = cal_center(val_loader, args, s_model)
    for fuck, (x, y, k) in enumerate(train_loader):
    
        s_model.train()
    
        x = x.cuda()
        y = y.cuda()
        k = k.cuda()
        with torch.no_grad():
            s_feats, logits = s_model(x, is_feat=True)
        probs = F.softmax(logits, dim=1)
        
        # confidence
        conf = probs.max(dim=1)[0]
        # margin
        rank = torch.argsort(probs, dim=1)
        top2 = torch.gather(probs, dim=1, index=rank[:,-2:])
        margin = top2[:,-1] - top2[:,-2]
        # entropy
        entropy = -torch.sum(probs * torch.log(probs), dim=1)

        if args.strategy == 0:
            scores = torch.rand(x.size(0)).cuda()
        elif args.strategy == 1:
            scores = 1 - conf
        elif args.strategy == 2:
            scores = -margin
        elif args.strategy == 3:
            scores = entropy
        else: 
            raise ValueError('Invalid strategy.')
    
        r = torch.arange(x.size(0)).float()
        m = (2*args.b-1) / (2*args.batch_size)
        mask_proto = 1 / (1 + torch.exp(-args.w * (r/args.batch_size - m) ))
        mask_proto = mask_proto.cuda()

        lamb = np.random.beta(1, 1)
        mask = lamb * mask_proto.view(-1, 1, 1, 1)
        rank = torch.argsort(scores, descending=True)
    
        index = torch.randperm(x.size(0)).cuda()
        x = (1-mask) * x[rank] + mask * x[index]
        x = x[:args.k]
    
        counter[epoch, k[rank[:args.b]] ] += 1
    
        s_feats, s_logits = s_model(x, is_feat=True)
    
        with torch.no_grad():
            t_feats, t_logits = t_model(x, is_feat=True)
            ## for statistics
            t_probs = F.softmax(t_logits, dim=1)
            # confidence
            t_conf = t_probs.max(dim=1)[0]
            # margin
            t_rank = torch.argsort(t_probs, dim=1)
            t_top2 = torch.gather(t_probs, dim=1, index=t_rank[:,-2:])
            t_margin = t_top2[:,-1] - t_top2[:,-2]
            # entropy
            t_entropy = -torch.sum(t_probs * torch.log(t_probs), dim=1)
    
        # compute loss
        log_s_probs = F.log_softmax(s_logits / args.T, dim=1)
        t_probs = F.softmax(t_logits / args.T, dim=1)
    
        tmp = mask.squeeze()[:args.k]
        loss_ce = F.cross_entropy(s_logits, y[rank][:args.k], reduction='none') * (1-tmp) + \
                    F.cross_entropy(s_logits, y[index][:args.k], reduction='none') * tmp
        loss_kd = F.kl_div(log_s_probs, t_probs, reduction='batchmean') * args.T * args.T
        if args.other_distill is not None:
            loss_other = sum(criterion(s_feats[1:-1], t_feats[1:-1])) if args.other_distill == 'AT' \
                        else sum(criterion(s_feats[-2], t_feats[-2]))
            loss = args.ce_weight * loss_ce.mean() + args.kd_weight * loss_kd + weight * loss_other
        else:
            loss = args.ce_weight * loss_ce.mean() + args.kd_weight * loss_kd
    
        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # compute distance between samples and center
        C = center[y[rank[:args.k]]]
        S = s_feats[-1]
        D = torch.pow(C-S, 2).sum(dim=1).sqrt().mean()
        record['center_dist'].update(D.item(), rank[:args.k].size(0))
    
        batch_acc = accuracy(logits, y, topk=(1,))[0]
        record['acc'].update(batch_acc.item(), logits.size(0))
        record['loss'].update(loss.item(), s_logits.size(0))

        i = rank[:args.k].size(0)
        record['s_select_confidence'].update(conf[rank[:args.k]].mean().item(), i)
        record['s_select_margin'].update(margin[rank[:args.k]].mean().item(), i)
        record['s_select_entropy'].update(entropy[rank[:args.k]].mean().item(), i)

        i = rank[args.k:].size(0)
        if i > 0:
            record['s_else_confidence'].update(conf[rank[args.k:]].mean().item(), i)
            record['s_else_margin'].update(margin[rank[args.k:]].mean().item(), i)
            record['s_else_entropy'].update(entropy[rank[args.k:]].mean().item(), i)

        i = conf.size(0)
        record['s_all_confidence'].update(conf.mean().item(), i)
        record['s_all_margin'].update(margin.mean().item(), i)
        record['s_all_entropy'].update(entropy.mean().item(), i)

        i = t_conf.size(0)
        record['t_confidence'].update(t_conf.mean().item(), i)
        record['t_margin'].update(t_margin.mean().item(), i)
        record['t_entropy'].update(t_entropy.mean().item(), i)
    
    for item in items:
        logger.add_scalar('train/{}'.format(item), record[item].avg, epoch+1)

    # val
    acc = val(val_loader, args, t_model, s_model, logger, epoch)

    if acc > best_acc:
        best_acc = acc
        state_dict = dict(state_dict=s_model.state_dict(), best_acc=best_acc)
        name = osp.join(exp_path, 'ckpt/student_best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
    
    scheduler.step()

if args.seed ==0 :
    counter = counter.cpu().numpy()
    np.save(osp.join(exp_path, 'counter.npy'), counter)
