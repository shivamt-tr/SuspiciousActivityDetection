# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:36:10 2020

@author: tripa
"""

import cv2
import numpy as np

def prepare(img):
    '''
    Returns image in the suitable format for training data input
    '''
    img = cv2.resize(img, (224, 224))
    # Convert the image to suitable format for model input
    img_convert = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

    return img_convert.reshape(-1, 224, 224, 3)


# %%

def binarizer(ar):
    output = np.zeros((2), np.float)
    output[ar] = 1
    return output
