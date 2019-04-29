"""
    This module contains tranforms definitions for training, validating and tesing stages.
"""

import cv2
import numpy as np

from albumentations import (
    Normalize,
    PadIfNeeded,
    Compose,
)

def train_transformations(prob=1.0):
    return Compose([PadIfNeeded(min_height=384, min_width=1248, always_apply=True), 
        Normalize(always_apply=True)], p=prob)

def valid_tranformations(prob=1.0):
    return Compose([PadIfNeeded(min_height=384, min_width=1248, value=(0,0,0), always_apply=True),
        Normalize(always_apply=True)], p=prob)
 
def test_trasformations(prob=1.0):
    pass