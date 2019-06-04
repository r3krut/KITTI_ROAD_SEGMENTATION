"""
    This module contains implementations of some activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FTSwishPlus(nn.Module):
    """
        Implementation of FTSwish+ activation function from https://arxiv.org/pdf/1812.06247.pdf
        Part of this code was taken from https://medium.com/@lessw/comparison-of-activation-functions-for-deep-learning-initial-winner-ftswish-f13e2621847
    """
    def __init__(self, threshold=-0.25, mean_shift=-0.1):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):

        #FTSwish+ for positive values
        pos_value = (x * F.sigmoid(x)) + self.threshold

        #FTSwish+ for negative values
        tval = torch.tensor([self.threshold], device=self.device)

        x = torch.where(x >= 0, pos_value, tval)

        if self.mean_shift is not None:
            x.sub_(self.mean_shift)

        return x