"""
    This module represents KITTI dataset.
    Key features:
        1:  Loading images and their coresponding masks for training or validation
        2:  Loading images and their coresponding names for test
"""

#general imports
import cv2
import numpy as np
from pathlib import Path

#torch packages imports
import torch
from torch.utils.data import Dataset, DataLoader

class RoadDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, is_valid=False, is_test=False):
        """
            params:
                dataset_path    : Path to the KITTI dataset which contains two subdirs: 'testing' and 'training'
                transforms      : Transformations for image augmentation
                is_valid        : Validation dataset loading
                is_test         : Loading only images without labels if 'True', else loading images and their labels
        """
        super().__init__()
        
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.is_valid = is_valid
        self.is_test = is_test

        if self.is_test:
            self.images_paths = sorted(list((self.dataset_path / 'testing' / 'image_2').glob('*')))
            self.labels_paths = None
        else:
            self.images_paths = sorted(list((self.dataset_path / 'training' / 'image_2').glob('*')))
            self.valid_labels_paths = sorted(list((self.dataset_path / 'training' / 'valid_masks').glob('*')))
            self.train_labels_paths = sorted(list((self.dataset_path / 'training' / 'train_masks').glob('*')))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if self.is_test:
            img = load_image_(str(self.images_paths[idx]))
            return img, str(self.images_paths[idx])
        else:
            if self.is_valid:
                img = load_image_(str(self.images_paths[idx]))
                valid_mask = load_mask_(str(self.valid_labels_paths[idx]))
                train_mask = load_mask_(str(self.train_labels_paths[idx]))
                img = img * np.expand_dims((train_mask == 0), axis=2)
                if self.transforms:
                    img, valid_mask = self.transforms(img, valid_mask)
                return img, valid_mask
            else:
                img = load_image_(str(self.images_paths[idx]))
                valid_mask = load_mask_(str(self.valid_labels_paths[idx]))
                train_mask = load_mask_(str(self.train_labels_paths[idx]))
                img = img * np.expand_dims((valid_mask == 0), axis=2)
                if self.transforms:
                    img, train_mask = self.transforms(img, train_mask)
                return img, train_mask

def load_image_(img_path):
    img = cv2.imread(img_path, 1)
    return img
    
def load_mask_(mask_path):
    mask = cv2.imread(mask_path, 0).astype(dtype=np.float32)/255
    return mask