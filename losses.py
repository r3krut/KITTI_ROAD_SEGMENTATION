"""
    This module contains loss definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEHaccardLoss:
    """
        This loss consists from two parts: (1 - alpha) * BCE - alpha * log(Jaccard)
    """
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, targets, outputs):
        loss = (1 - self.alpha) * self.bce_loss(outputs, targets)

        if self.alpha:
            epsilon = 1e-15

            jaccard_targets = (targets == 1).float()
            jaccard_outputs = F.sigmoid(outputs)

            intersection = (jaccard_targets * jaccard_outputs).sum()
            union = jaccard_targets.sum() + jaccard_outputs.sum()

            loss -= self.alpha * torch.log((intersection + epsilon) / (union - intersection + epsilon))
        return loss