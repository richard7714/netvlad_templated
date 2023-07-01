# 사용할 loss 목록을 넣어놓으면 될듯?
import torch
import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def triplet_loss(anchor,positive,negative,margin):
    criterion = nn.TripletMarginLoss(margin=margin**0.5,
                p = 2, reduction='sum')
    
    loss = criterion(anchor,positive,negative)
    
    return loss
    