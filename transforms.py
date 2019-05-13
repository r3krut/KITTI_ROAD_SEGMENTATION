"""
    This module contains tranforms definitions for training, validating and tesing stages.
"""

import cv2
import numpy as np

from albumentations import (
    HorizontalFlip,
    PadIfNeeded,
    RandomCrop,
    Normalize,
    Resize,
    Rotate,
    ToGray,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    CLAHE,
    Blur,
    OneOf,
    Compose,
)


def train_transformations(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=384, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
        # RandomCrop(height=256, width=256, always_apply=True),
        OneOf([HorizontalFlip(p=0.5), Rotate(limit=30, p=0.5)], p=0.5),
        OneOf([ToGray(p=0.5), 
            RandomBrightness(p=0.5), 
            RandomContrast(p=0.5), 
            RandomGamma(p=0.5), 
            CLAHE(p=0.5)], p=0.5),
        Normalize(always_apply=True)], p=prob)


def valid_tranformations(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=384, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
        # RandomCrop(height=256, width=256, always_apply=True),
        Normalize(always_apply=True)], p=prob)
 

def test_trasformations(prob=1.0):
    pass