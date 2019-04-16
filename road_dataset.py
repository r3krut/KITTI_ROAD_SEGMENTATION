"""
    This module represents KITTI dataset.
    Key features:
        1:  Loading images and their coresponding masks 
"""

#general imports
import cv2
import numpy as np
from pathlib import Path

#torch packages imports
import torch
from torch.utils.data import Dataset, DataLoader

class RoadDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, is_test=False):
        """
            params:
                dataset_path    : Path to the KITTI dataset which contains two subdirs: 'testing' and 'training'
                transforms      : Transformations for image augmentation
                is_test         : Loading only images without labels if 'True', else loading images and their labels
        """
        super().__init__()
        
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.is_test = is_test

        if self.is_test:
            self.images_paths = list((self.dataset_path / 'testing' / 'image_2').glob('*'))
            self.labels_paths = None
        else:
            self.images_paths = list((self.dataset_path / 'training' / 'image_2').glob('*'))
            self.labels_paths = list((self.dataset_path / 'training' / 'gt_image_2').glob('*'))    

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        pass

    def load_image_(self, img_path):
        pass
    
    def load_mask_(self, mask_path):
        pass