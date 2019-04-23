"""
    Module for image utils
"""

import cv2
import numpy as np
from pathlib import Path

class ImageSpecifications:
    """
        Class that represents specifications of an image
    """
    imageShape_max = (376, 1242)
    img_extension = ".png"


def alpha_overlay(img, gt_image, color=(0, 255, 0), alpha=0.5):
    """
        Method for visualizing mask above image. Part of this code was taken from https://github.com/ternaus/TernausNet/blob/master/Example.ipynb

        params:
            img         : source image
            gt_image    : ground truth image(mask)
            color       : mask color
            alpha       : overlaing parameter in equation: (1-alpha)*img + alpha*gt_image
    """
    #assert len(gt_image.shape) == len(img.shape), "Error: Ground truth and source images has different shapes."
    #assert gt_image.shape[0] == 1, "Error: Ground truth image has channels more then 1."

    gt_image = np.dstack((gt_image, gt_image, gt_image)) * np.array(color)
    gt_image = gt_image.astype(np.uint8)
    weighted_sum = cv2.addWeighted(gt_image, (1-alpha), img, alpha, 0.)
    img2 = img.copy()

    if color == (255, 0, 0):
        channel_pos = 0
    elif color == (0, 255, 0):
        channel_pos = 1
    elif color == (0, 0, 255):
        channel_pos = 2
    else:
        raise ValueError("Wrong color: {}. Color should be 'red', 'green' or 'blue'.".format(color))
    
    ind = gt_image[:, :, channel_pos] > 0     
    img2[ind] = weighted_sum[ind] 
    return img2


def pad(img, required_size=ImageSpecifications.imageShape_max, background_value=0):
    """
        Padding for ground truth images
        
        params:
            img                 : image for padding
            required_size       : size after padding
            background_value    : 1 if 'img' is validation map then else 0
    """
    
    assert len(required_size) == 2, "required_size dimmention isn't equals 2."
    assert img.shape[0] <= required_size[0], "height of image greater then height of required_size."
    assert img.shape[1] <= required_size[1], "width of image greater then width of required_size."

    if (img.shape[0] == required_size[0]) and (img.shape[1] == required_size[1]):
        return img

    if background_value == 0:
        new_img = np.zeros(required_size, dtype=img.dtype)
        new_img[:img.shape[0], :img.shape[1]] = img
        return new_img
    elif background_value == 1:
        new_img = np.ones(required_size, dtype=img.dtype) * 255
        new_img[:img.shape[0], :img.shape[1]] = img
        return new_img
    else:
        raise ValueError("\'background_value\' is not valid: {}".format(background_value))


def getGroundTruth(fileNameGT, make_pad=False):
    """
        Returns the ground truth maps for roadArea and the validArea 
        
        param:
            fileNameGT  : ground truth file name
            make_pad    : 
    """
    # Read GT   
    assert fileNameGT.is_file(), "Cannot find: {}".format(fileNameGT)
    full_gt = cv2.imread(str(fileNameGT), 1)

    #attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea = full_gt[:,:,0] > 0
    validArea = full_gt[:,:,2] > 0

    if not make_pad:
        return roadArea, validArea

    roadArea = roadArea.astype(dtype=np.uint8) * 255
    validArea = validArea.astype(dtype=np.uint8) * 255

    roadArea = pad(roadArea)
    validArea = pad(validArea, background_value=1)

    roadArea = (roadArea/255).astype(dtype=np.bool)
    validArea = (validArea/255).astype(dtype=np.bool)

    return roadArea, validArea