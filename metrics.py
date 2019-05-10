"""
    This module contains some metrics
"""

import torch
import numpy as np


def batch_jaccard(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + predict.sum(dim=-2).sum(dim=-1)
    jaccard = (intersection + eps) / (union - intersection + eps)
    return list(jaccard.data.cpu().numpy())


def batch_dice(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + predict.sum(dim=-2).sum(dim=-1)
    dice = (2 * intersection + eps) / (union + eps)
    return list(dice.data.cpu().numpy())


def jaccard(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum().float()
    union = target.sum().float() + predict.sum().float()
    return ((intersection + eps) / (union - intersection + eps)).cpu().item()


def dice(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum()
    union = target.sum() + predict.sum()
    return ((2 * intersection + eps) / (union + eps)).cpu().item()