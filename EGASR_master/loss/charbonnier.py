import torch
import torch.nn as nn
import torch.nn.functional as F


class Charbonnier(nn.Module):
    def __init__(self):
        super(Charbonnier, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss
