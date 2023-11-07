#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:37:27 2023

@author: lucas
"""

import cv2
import numpy as np
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
from quantify_segmentation import get_props_per_cell, detect_big_cells, detect_not_rounded_cells, get_img_from_idx_cells, list_cells
from aux_functions.functionPercNorm import functionPercNorm, get_one_channel

###################################   PARAMETERS   #########################

image_input         = ''
path_model_trained  = ''
flag_normalize = False
flag_gpu = False
channels = [[0,0]] #Same channels as training
th_size=150

#Change if not desired naming
csv_output = image_input + '_descriptors.csv'
segmentation_output = image_input + '_seg.png'

##############################################################################


def main():
    
    #Load model
    model_trained = models.CellposeModel(pretrained_model=path_model_trained, gpu=flag_gpu)
    
    #Load image (first channel)
    img_original = imread(image_input)
    img_original = get_one_channel(img_original)
    
    #Load image (first channel)
    if flag_normalize:
        img_original = functionPercNorm( np.single(img_original))
    
    #Segment image
    img_segmentation, flows, styles = model_trained.eval(img_original, diameter=None, channels= channels)
    
    #Get centroids and shape descriptors
    cell_props = get_props_per_cell(img_segmentation)
    
    #Get not compact cells
    vector_compactness_per_cell, not_compact_idx = detect_not_rounded_cells(cell_props)
    img_not_rounded_cells = get_img_from_idx_cells(img_segmentation, not_compact_idx)
    
    #Get big cells
    vector_area_per_cell, big_cells_idx = detect_big_cells(cell_props, th_size=th_size)
    img_big_cells = get_img_from_idx_cells(img_segmentation, big_cells_idx)
    
    #Save CSV
    list_cells(cell_props, csv_output)
    
    #Save segmentation
    cv2.imwrite(segmentation_output, img_segmentation)
    
    #Plot
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    
    plt.subplot(2, 4, 1)
    plt.imshow(img_original,cmap='gray')
    plt.gca().set_title('Original')
    plt.subplot(2, 4, 2)
    plt.imshow(img_segmentation)
    plt.gca().set_title('Segmentation - Total cells:' + str(len(cell_props)))
    
    plt.subplot(2, 4, 3)
    plt.hist(vector_compactness_per_cell, bins=20)
    plt.gca().set_title('Hist compactness')
    plt.subplot(2, 4, 4)
    plt.imshow(img_not_rounded_cells)
    plt.gca().set_title('Not compact cells: ' + str(len(not_compact_idx)))
    
    plt.subplot(2, 4, 5)
    plt.hist(vector_area_per_cell, bins=20)
    plt.gca().set_title('Hist area')
    plt.subplot(2, 4, 6)
    plt.imshow(img_big_cells)
    plt.gca().set_title('Big cells: ' + str(len(big_cells_idx)))
    
    
    figManager = plt.get_current_fig_manager()
    
    figManager.window.showMaximized()
    
    plt.show()
    
    
if __name__ == "__main__":
    main()