"""
    This code contains implementation of the Polynominal Learnning Rate scheduler.
"""

import math
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class PolyLR(_LRScheduler):
    """
        This is poly LR scheduler that changes initial LR every step in the following manner: n_next(i) = n_initial(1-i/N)^alpha
    """
    def __init__(self, optimizer, num_epochs=100, alpha=0.9, last_epoch=-1):
        """
            Params:
                num_epochs      : number of train epochs
                alpha           : alpha parameter
        """
        self.num_epochs = num_epochs
        self.alpha = alpha

        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * math.pow((1-self.last_epoch/self.num_epochs), self.alpha) for base_lr in self.base_lrs]