# 사용할 loss 목록을 넣어놓으면 될듯?
import torch
import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def triplet_loss(output, target):
    criterion = nn.TripletMarginLoss(margin=0.1**0.5,
                p = 2, reduction='sum')
    return criterion
    