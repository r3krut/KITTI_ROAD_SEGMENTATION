"""
    This module perform preparation of the train and validation datasets for KITTI benchmark
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

from img_utils import (
        getGroundTruth,
        ImageSpecifications,
)

train_masks_dir = "train_masks"
valid_masks_dir = "valid_masks"
gt_image_2_dir = "gt_image_2"                           #   Ground truth train and validation masks 
image_2_dir = "image_2"                                 #   RGB images
droped_valid_image_2_dir = "droped_valid_image_2"       #   source train images without validation areas

def prepare_train_valid(path2train):
    """
        Select train and validation masks and save it in separate dirs.
        Params:
            path2train: path to a dir which contains 'gt_image' and 'image_2' subdirs
    """

    path2train = Path(path2train)

    if not path2train.exists():
        raise ValueError("Dir '{}' is not exist.".format(path2train))

    gt_image_2 = path2train / gt_image_2_dir
    image_2 = path2train / image_2_dir

    images_path_list = sorted(list(image_2.glob('*')))
    gt_images_path_list = sorted(list(gt_image_2.glob('*')))

    #Deletion of lane examples 
    gt_images_path_list = gt_images_path_list[95:]

    assert len(images_path_list) == len(gt_images_path_list), "Error: 'images_path_list' and 'gt_images_path_list' has different sizes."

    #Train mask dir
    train_masks = path2train / train_masks_dir
    train_masks.mkdir(parents=True, exist_ok=True)

    #Valid mask dir
    valid_masks = path2train / valid_masks_dir
    valid_masks.mkdir(parents=True, exist_ok=True)

    for idx, img_name in enumerate(images_path_list):
        img_name = str(img_name)
        print("Processing '{}' file.".format(img_name))

        db_name = img_name.split('/')[-1].split('.')[0].split('_')[0]
        num = img_name.split('/')[-1].split('.')[0].split('_')[1]

        road_maks, valid_mask = getGroundTruth(gt_images_path_list[idx])
        
        road_maks = road_maks.astype(dtype=np.uint8) * 255
        valid_mask = (~valid_mask).astype(dtype=np.uint8) * 255
    
        cv2.imwrite(str(train_masks / (db_name + '_' + num + ImageSpecifications.img_extension)), road_maks)
        cv2.imwrite(str(valid_masks / (db_name + '_' + num + ImageSpecifications.img_extension)), valid_mask)


def drop_valid(path2train):
    """
        Drops validation areas from original train images.
        Params:
            path2train: path that contains 'image_2' and 'valid_masks' subdirs
    """

    path2train = Path(path2train)

    droped_valid_image_2 = path2train / droped_valid_image_2_dir
    droped_valid_image_2.mkdir(parents=True, exist_ok=True)

    src_images_paths = sorted(list((path2train / image_2_dir).glob("*")))
    valid_masks_paths = sorted(list((path2train / valid_masks_dir).glob("*")))

    assert len(src_images_paths) == len(valid_masks_paths), "Error: src_images_paths and valid_masks_paths has different length."

    for i in range(0, len(src_images_paths)):
        src_image = cv2.imread(str(src_images_paths[i]))
        valid_mask = cv2.imread(str(valid_masks_paths[i]), 0)
        src_image = src_image * np.expand_dims((valid_mask == 0), axis=2).astype(dtype=np.uint8)
        
        img_name = str(src_images_paths[i]).split("/")[-1]        
        cv2.imwrite(str(droped_valid_image_2 / img_name), src_image)
        print("Image {0} porcessed.".format(str(src_images_paths[i])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Argument parser for data_processing module")
    ap.add_argument("--train-dir", type=str, required=True, help="Path to a train dir(should contain 'gt_image_2' and 'image_2' subdirs).")
    args = ap.parse_args()

    #prepare_train_valid(args.train_dir)
    drop_valid(args.train_dir)

    print("Done!")