# 사용할 loss 목록을 넣어놓으면 될듯?

import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)
