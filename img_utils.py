"""
    Module for image utils
"""

import cv2
import numpy as np

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
