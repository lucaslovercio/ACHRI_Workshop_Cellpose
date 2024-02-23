#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:47:06 2024

@author: lucas
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose.io import imread
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from quantify_segmentation import get_correspondance_segmentations
from aux_functions.functionPercNorm import functionPercNorm, get_one_channel


###################################   PARAMETERS   #########################
folder = ''
input_nuclei    = folder + 'dapi_5.jpg'
input_membrane  = folder + 'membrane_5.jpg'

#Only to re-compute for visualization. It does not produce the files again
min_pixels_matching = 1500
csv_input                      = 'descriptors'
segmentation_nuclei_input        = input_nuclei + '_seg.png'
segmentation_membrane_input    = input_membrane + '_seg.png'

##############################################################################


def figure_segmentation_matching_results(img_membrane_original, img_segmentation_membrane, img_nuclei_original,\
                                         img_segmentation_nuclei,img_segmentation_membrane_multiple_nuclei, img_segmentation_nuclei_in_same_cells,\
                                             img_segmentation_nuclei_sharing_cells, img_segmentation_membranes_sharing_nuclei):
    
    #Image composition
    img_composition = (img_nuclei_original / 2) + (img_membrane_original / 2)
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    #fig = plt.figure()
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    
    plt.subplot(3, 3, 1)
    plt.imshow(img_membrane_original,cmap='gray')
    plt.gca().set_title('Original')
    plt.subplot(3, 3, 2)
    plt.imshow(img_segmentation_membrane,cmap='gist_ncar')
    plt.gca().set_title('Segmentation Cells')
    
    plt.subplot(3, 3, 3)
    plt.imshow(img_nuclei_original,cmap='gray')
    plt.gca().set_title('Original')
    plt.subplot(3, 3, 4)
    plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
    plt.gca().set_title('Segmentation Nuclei:')
    
    plt.subplot(3, 3, 5)
    plt.imshow(img_composition,cmap='gray')
    plt.gca().set_title('Composition')
    
    plt.subplot(3, 3, 6)
    plt.imshow(img_segmentation_membrane_multiple_nuclei,cmap='gist_ncar')
    plt.gca().set_title('Multi nuclei cells')
    
    plt.subplot(3, 3, 7)
    plt.imshow(img_segmentation_nuclei_in_same_cells,cmap='gist_ncar')
    plt.gca().set_title('Nuclei sharing a cell')
    
    
    plt.subplot(3, 3, 8)
    plt.imshow(img_segmentation_nuclei_sharing_cells,cmap='gist_ncar')
    plt.gca().set_title('Nuclei in more than a cell - data cleaning')
    
    plt.subplot(3, 3, 9)
    plt.imshow(img_segmentation_membranes_sharing_nuclei,cmap='gist_ncar')
    plt.gca().set_title('Cells sharing nuclei - data cleaning')
    
    figManager = plt.get_current_fig_manager()
    
    figManager.window.showMaximized()
    
    plt.show()


def main():
    
    
        #Load image (first channel)
    img_membrane_original = imread(input_membrane)
    if isinstance(img_membrane_original, type(None)):
        print("---------------------------------------")
        print("---- File of membrane not found!!!!----")
        print("---- See path below -------------------")
        print("---------------------------------------")
        sys.exit(1)
            
    img_membrane_original = get_one_channel(img_membrane_original)
    #Normalize anyway, this script if just for visualization
    img_membrane_original = functionPercNorm( np.single(img_membrane_original))
    
    #Load image (first channel)
    img_nuclei_original = imread(input_nuclei)
        
    if isinstance(img_nuclei_original, type(None)):
        print("--------------------------------------")
        print("---- File of nuclei not found!!!!-----")
        print("---- See path below ------------------")
        print("--------------------------------------")
        sys.exit(1)

    img_nuclei_original = get_one_channel(img_nuclei_original)
    #Normalize anyway, this script if just for visualization
    img_nuclei_original = functionPercNorm( np.single(img_nuclei_original))
        
    #Load nuclei segmentation
    img_segmentation_nuclei = imread(segmentation_nuclei_input)
    #Load membrane segmentation
    img_segmentation_membrane = imread(segmentation_membrane_input)
        
    if isinstance(img_segmentation_nuclei, type(None)) or isinstance(img_segmentation_membrane, type(None)):
        print("--------------------------------------------")
        print("---- File of a segmentation not found!!!!----")
        print("----------- See path below ------------------")
        print("--------------------------------------------")
        sys.exit(1)
    
    img_segmentation_membrane_multiple_nuclei, img_segmentation_nuclei_in_same_cells, labels_multinuclei_cells =\
        get_correspondance_segmentations(img_segmentation_membrane, img_segmentation_nuclei, min_pixels_matching=min_pixels_matching)
        
    img_segmentation_nuclei_sharing_cells, img_segmentation_membranes_sharing_nuclei, labels_nucleus_sharing_cells =\
        get_correspondance_segmentations(img_segmentation_nuclei, img_segmentation_membrane, min_pixels_matching=min_pixels_matching)
    
    figure_segmentation_matching_results(img_membrane_original, img_segmentation_membrane, img_nuclei_original,\
                                         img_segmentation_nuclei,img_segmentation_membrane_multiple_nuclei, img_segmentation_nuclei_in_same_cells,\
                                             img_segmentation_nuclei_sharing_cells, img_segmentation_membranes_sharing_nuclei)
    
    
if __name__ == "__main__":
    main()