"""
    This module contains utils for validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils import to_gpu

def validation(model: nn.Module, criterion, valid_loader):
    with torch.no_grad():
        valid_losses = []
        jaccards = []
        dices = []
        
        model.eval()
        for idx, batch in enumerate(valid_loader):
            inputs, targets = batch
            inputs = to_gpu(inputs)
            targets = to_gpu(targets)
            outputs = model(inputs)

            loss = criterion(targets, outputs)
            valid_losses.append(loss)
            jaccards += calc_jaccard(targets, outputs)
            dices += calc_dice(targets, outputs)

        #Calculates losses
        valid_loss = np.mean(valid_losses)
        valid_jaccard = np.mean(jaccards).astype(dtype=np.float64)  
        valid_dice = np.mean(dices).astype(dtype=np.float64)      
        
        #print("Validation loss: {0}, Validation Jaccard: {1}, Validation DICE: {2}".format(valid_loss, valid_jaccard, valid_dice))
        return {"val_loss": valid_loss, "val_jacc": valid_jaccard, "val_dice": valid_dice}


def calc_jaccard(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + predict.sum(dim=-2).sum(dim=-1)
    jaccard = (intersection + eps) / (union - intersection + eps)
    return list(jaccard.data.cpu().numpy())


def calc_dice(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + predict.sum(dim=-2).sum(dim=-1)
    dice = 2 * (intersection + eps) / (union + eps)
    return list(dice.data.cpu().numpy())