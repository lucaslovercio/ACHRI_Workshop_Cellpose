#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:35:08 2024

@author: lucas
"""

###################################   PARAMETERS   #########################
tiff_path = '' #In Windows, place an r before the ''

#Names must be the same used for the processing
label_x = 'Channel 2'
label_y = 'Channel 3'
segmentation_dapi_nuclei_output        = tiff_path + '_nuclei_seg.png'
segmentation_channel2_output        = tiff_path + '_'+ label_x +'_seg.png'
segmentation_channel3_output        = tiff_path + '_'+ label_y +'_seg.png'
n_channels = 3

###################################   PARAMETERS   #########################


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose.io import imread
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list, split_list_images, get_projected_image
from quantify_segmentation import get_props_per_cell

#Load dapi segmentation
img_segmentation_nuclei = imread(segmentation_dapi_nuclei_output)
if isinstance(img_segmentation_nuclei, type(None)):
    print("---------------------------------------")
    print("---- File of DAPI seg not found!!!!----")
    print("---- See path below -------------------")
    print("---------------------------------------")
    sys.exit(1)
    
#Load dapi segmentation
img_segmentation_channel2 = imread(segmentation_channel2_output)
if isinstance(img_segmentation_channel2, type(None)):
    print("---------------------------------------")
    print("---- File of Channel 2 Seg not found!!!!----")
    print("---- See path below -------------------")
    print("---------------------------------------")
    sys.exit(1)
    
#Load dapi segmentation
img_segmentation_channel3 = imread(segmentation_channel3_output)
if isinstance(img_segmentation_channel3, type(None)):
    print("---------------------------------------")
    print("---- File of Channel 2 Seg not found!!!!----")
    print("---- See path below -------------------")
    print("---------------------------------------")
    sys.exit(1)
    
images = read_multipage_tiff_as_list(tiff_path)

if isinstance(images, type(None)):
    print("---------------------------------------")
    print("---- Original TIFF File not found!!!!----")
    print("---- See path below -------------------")
    print("---------------------------------------")
    sys.exit(1)

list_list_images = split_list_images(images, n_channels)

#Split
#Numbering of variables starting in 1 for reader/user of the code
img_channel1_original = get_projected_image(list_list_images[0])
img_channel2_original = get_projected_image(list_list_images[1])
img_channel3_original = get_projected_image(list_list_images[2])

cell_props_nuclei = get_props_per_cell(img_segmentation_nuclei)
cell_props_channel2 = get_props_per_cell(img_segmentation_channel2)
cell_props_channel3 = get_props_per_cell(img_segmentation_channel3)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
#fig = plt.figure()
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)

plt.subplot(2, 4, 1)
plt.imshow(img_channel1_original,cmap='gray')
plt.gca().set_title('Nuclei channel')
plt.subplot(2, 4, 2)
plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
plt.gca().set_title('Nuclei: ' + str(len(cell_props_nuclei)))

plt.subplot(2, 4, 3)
plt.imshow(img_channel2_original,cmap='gray')
plt.gca().set_title(label_x)
plt.subplot(2, 4, 4)
plt.imshow(img_segmentation_channel2,cmap='gist_ncar')
plt.gca().set_title(label_x + ': ' + str(len(cell_props_channel2)))

plt.subplot(2, 4, 5)
plt.imshow(img_channel3_original,cmap='gray')
plt.gca().set_title(label_y)
plt.subplot(2, 4, 6)
plt.imshow(img_segmentation_channel3,cmap='gist_ncar')
plt.gca().set_title(label_y + ': ' + str(len(cell_props_channel3)))

figManager = plt.get_current_fig_manager()

#figManager.window.showMaximized()

plt.show()
