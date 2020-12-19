from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

