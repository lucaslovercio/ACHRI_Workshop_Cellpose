#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import ndimage
from PIL import Image
import numpy as np
import os
import imageio


"""
Created on Thu Sep 14 16:53:27 2023

@author: lucas
"""

def function_enconde_annotation_from_binary_mask(fullpath_orig, filename, folder_dest):
    img = Image.open(fullpath_orig)
 
    numpydata = np.asarray(img)
    
    shape_img = np.shape(numpydata)
    print(shape_img)
    print(np.size(shape_img))
    if np.size(shape_img)>2:
        mask = numpydata[:, :, 0] > 0
    else:
        mask = numpydata > 0
    
    labels, nb = ndimage.label(mask)
    
    fullpath_dest = os.path.join(folder_dest,filename+'_masks.png')
    mask_encoded = Image.fromarray(labels.astype(np.uint16))
    mask_encoded.save(fullpath_dest)

def function_enconde_annotation(fullpath_orig, filename, folder_dest):
    img = Image.open(fullpath_orig)
 
    numpydata = np.asarray(img)
    
    #Coloured pixels do not have same value in the three channels
    mask = (numpydata[:, :, 0] != numpydata[:, :, 1]) | (numpydata[:, :, 0] != numpydata[:, :, 2])
    
    labels, nb = ndimage.label(mask)
    
    fullpath_dest = os.path.join(folder_dest,filename+'_masks.png')
    mask_encoded = Image.fromarray(labels.astype(np.uint16))
    mask_encoded.save(fullpath_dest)
    
    #In case Pillow does not save in 16 bits, use imageio to save
    #imageio.imwrite(fullpath_dest,labels.astype(np.uint16)) 
    
def function_encode_annotations(folder_rgb_annotations, folder_masks, flag_binary_mask = False):
    if not os.path.exists(folder_masks):
        os.makedirs(folder_masks)
    
    path_images = []
    file_ending = ''
    for file in os.listdir(folder_rgb_annotations):
        if file.endswith(".png") or file.endswith(".bmp")  or file.endswith(".tif"):
            filename_tmp = file[0:-4]
            file_ending = file[-4:]
            path_images.append(filename_tmp)
    nFiles = len(path_images)
    
    for imgNumber in range(nFiles):
        fullpath_rgb = os.path.join(folder_rgb_annotations, path_images[imgNumber] + file_ending)
        if flag_binary_mask:
            function_enconde_annotation_from_binary_mask(fullpath_orig=fullpath_rgb, filename=path_images[imgNumber], folder_dest=folder_masks)
        else:
            function_enconde_annotation(fullpath_orig=fullpath_rgb, filename=path_images[imgNumber], folder_dest=folder_masks)
    