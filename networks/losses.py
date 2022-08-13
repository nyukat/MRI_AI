import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(BCELossWithSmoothing, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
    
    def forward(self, preds, label):
        label = label.float()
        if self.smoothing > 0.0:
            label = torch.abs(label - self.smoothing)
        return F.binary_cross_entropy_with_logits(preds, label)
    

def bce(preds, label):
    label = label.float()
    return F.binary_cross_entropy_with_logits(preds, label)
