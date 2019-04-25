"""
    This module contains models definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RekNetM1(nn.Module):
    """
        Simple baseline FCN model
    """
    def __init__(self, num_classes=1):
        super(RekNetM1, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x