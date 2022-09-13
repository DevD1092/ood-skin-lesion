import torch
import torch.nn as nn
import torch.nn.functional as F
from Dist import Dist

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

def mse_loss(pred, y_a, y_b, lam):
    return lam * (F.mse_loss(pred, y_a) / 2) + (1 - lam) * (F.mse_loss(pred, y_b) / 2)

class GCPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(GCPLoss, self).__init__()
        self.weight_pl = options['weight_pl']
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim']) # 

    def forward(self, x, y, labels=None, targets_a=None, targets_b=None, lam=None, mixup=0):
        dist = self.Dist(x)
        logits = F.softmax(-dist, dim=1)
        if labels is None: return logits, 0

        if mixup == 0:
            loss = F.cross_entropy(-dist / self.temp, labels)
            center_batch = self.Dist.centers[labels, :]
            loss_r = F.mse_loss(x, center_batch) / 2

        elif mixup == 1:
            loss = mixup_criterion(-dist, targets_a, targets_b, lam)
            center_batch_a = self.Dist.centers[targets_a, :]
            center_batch_b = self.Dist.centers[targets_b, :]
            loss_r = mse_loss(x, center_batch_a, center_batch_b, lam)

        loss = loss + self.weight_pl * loss_r
        return logits, loss