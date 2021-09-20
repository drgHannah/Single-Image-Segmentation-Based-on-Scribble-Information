import os
import math
import numpy as np

import torch 
import scipy
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


def color_map(N=256, normalized=False):
    ''' Generates a color map.
    
    This function is modified from the implementation of 
    https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_util/src/image_recognition_util

    '''

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):

        r = g = b = 0
        c = i+1
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_color_map_index():
    ''' Creates new colormap.
    '''
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    
    for i in range(255):
        newcolors[i, :] = np.append(color_map()[i]/255,1)
    newcmp = ListedColormap(newcolors)
    return newcmp

def blend_image_segmentation(image, target, amount=0.5):
    ''' Blend image with target, set segmentation to predefined colors.
    '''
    target = np.array(target)[:, :, np.newaxis]

    # apply colors to segmentation
    cmap = color_map()[:, np.newaxis, :]
    new_im = np.dot(target == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(target == i, cmap[i])

    # to pillow
    new_im = Image.fromarray(new_im.astype(np.uint8))
    image = Image.fromarray(np.array(image.permute(1,2,0)*255).astype(np.uint8))

    # blend
    blend_image = Image.blend(image, new_im, alpha=amount)
    return np.array(blend_image)/255.0


