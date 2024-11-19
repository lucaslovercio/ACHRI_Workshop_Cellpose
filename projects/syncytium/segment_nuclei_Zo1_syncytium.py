#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:22:46 2024

@author: lucas
"""

import os

###################################   PARAMETERS   #########################

pattern_channel_dapi = '_DAPI_'
pattern_channel_Zo1 = '_568_'

folder_images = ''
folder_models = ''

path_model_trained_level_01  = os.path.join(folder_models,'TSC_level_01_diam100_ji_0.692_cyto.137946')
path_model_trained_level_02  = os.path.join(folder_models,'TSC_level_02_cyto2_ji_0.493_diam_200.593621')
path_model_trained_level_03  = os.path.join(folder_models,'TSC_level_03_ji_0.54_diameter_100_cyto.502330')
path_model_trained_level_04  = os.path.join(folder_models,'TSC_level_04_diam100_nuclei_ji0.5928.142752')

flag_normalize = True
flag_gpu = True
channels = [[0,0]] #Same channels as training

# Parameters for 20x
diam_nuclei = 100
min_pixels_matching = 2000
th_size_noisy_cells = 1000
disk_diameter = 90
fontsize = 10

level_01_diam = None
level_02_diam = None
level_03_diam = None
level_04_diam = None

n_bins = 10

# Colours RGB of outputs

color_add_free_nuclei = [50, 0, 0]
color_add_back = [0, 40, 40]

color_edges_01 = [255, 17, 0]
color_edges_02 = [255, 250, 0]
color_edges_03 = [21, 250, 0]
color_edges_04 = [0, 40, 255]

color_rainbow_sum_1 = [0, 40, 255]
color_rainbow_sum_2 = [21, 250, 0]
color_rainbow_sum_3 = [255, 250, 0]
color_rainbow_sum_4 = [255, 17, 0]

color_add_rainbow_1 = [0, 0, 80]
color_add_rainbow_2 = [0, 80, 0]
color_add_rainbow_3 = [40, 40, 0]
color_add_rainbow_4 = [80, 0, 0]

color_nuclei = [0, 150, 150]
color_add_multiple_nuclei = [80, 80, 10]


##############################################################################
##############################################################################
##############################################################################
##############################################################################

import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
import numpy as np
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
from aux_functions.functionPercNorm import functionPercNorm, get_one_channel
from skimage.segmentation import expand_labels
from skimage.morphology import dilation, square
from skimage import filters
from quantify_segmentation import matching_label_pairs, get_correspondance_segmentations, get_img_from_idx_cells, \
    get_props_per_cell, get_areas, detect_big_cells, draw_roi_over_image, draw_mask_over_image, draw_mask_over_image_rgb, overlap_mask_over_image_rgb
from scipy import ndimage
from scipy.ndimage import label #, gaussian_filter
from skimage.morphology import disk
from aux_functions.filepath_management import get_sample_name, get_image_filenames
import cv2
from scipy import stats


def get_largest_empty_space(mask, structure_closing):
    
    mask_closed = ndimage.binary_closing(mask, structure=structure_closing, border_value=1)
    mask_closed = np.logical_not(mask_closed) #empty spaces are 0, as the background, and it will not be labelled. that is why it is inverted
    # Label the connected components for background detection
    labeled_array, num_regions = label(mask_closed)
    
    # Find the largest connected component
    largest_component = np.zeros_like(mask_closed)
    if num_regions > 0:
        component_sizes = [(labeled_array == i).sum() for i in range(1, num_regions + 2)]
        largest_component_index = np.argmax(component_sizes) +1 #because labels start from 1
        largest_component = (labeled_array == largest_component_index)
        
    return largest_component, labeled_array, mask_closed

def get_large_empty_spaces(mask, structure_closing, th_size_px = 80000):
    
    large_spaces = np.zeros_like(mask) == 1 #To have a boolean matrix
    mask_closed = ndimage.binary_closing(mask, structure=structure_closing, border_value=1)
    mask_closed = np.logical_not(mask_closed) #empty spaces are 0, as the background, and it will not be labelled. that is why it is inverted
    # Label the connected components for background detection
    labeled_array, num_regions = label(mask_closed)
        
    if num_regions > 0:
        component_sizes = [(labeled_array == i).sum() for i in range(0, num_regions + 2)]
        
        for j in range(1, len(component_sizes)):
            if component_sizes[j]>th_size_px:
                large_spaces[labeled_array==j] = True
            
    return large_spaces, labeled_array, mask_closed


image_filenames = get_image_filenames(folder_images, substring_pattern = pattern_channel_dapi)
for filename_nuclei in image_filenames:
    plt.close('all')
    print('------------------------------------')
    print('Reading: ' + filename_nuclei)    
    
    sample_name = get_sample_name(filename_nuclei)
    folder_output = os.path.join(folder_images, sample_name)
    folder_output_intermediate_output = os.path.join(folder_output, 'intermediate_output')
    
    filename_nuclei_Zo1 = filename_nuclei.replace(pattern_channel_dapi, pattern_channel_Zo1)
    
    image_input_nuclei    = os.path.join(folder_images,filename_nuclei)
    image_input_Zo1  = os.path.join(folder_images,filename_nuclei_Zo1)
    
    #Load image (first channel)
    img_original_Zo1 = imread(image_input_Zo1)
    img_original_nuclei = imread(image_input_nuclei)
    
    
    if (img_original_Zo1 is None) or (img_original_nuclei is None):
        print('Failed to load ' + sample_name)
    else:
        print('Analyzing: ' + filename_nuclei)
        os.makedirs(folder_output, exist_ok=True)
        os.makedirs(folder_output_intermediate_output, exist_ok=True)
        
        img_original_Zo1 = get_one_channel(img_original_Zo1)
        img_original_nuclei = get_one_channel(img_original_nuclei)
        
        #Load image (first channel)
        if flag_normalize:
            img_original_Zo1 = functionPercNorm( np.single(img_original_Zo1))
            img_original_nuclei = functionPercNorm( np.single(img_original_nuclei))
        
        
        #Load model 01
        model_trained_01 = models.CellposeModel(pretrained_model=path_model_trained_level_01, gpu=flag_gpu)
        #Segment image
        img_segmentation_01, flows, styles = model_trained_01.eval(img_original_Zo1, diameter=level_01_diam, channels= channels)
        del model_trained_01
        img_segmentation_01 = np.uint16(img_segmentation_01)
        
        #Load model 02
        model_trained_02 = models.CellposeModel(pretrained_model=path_model_trained_level_02, gpu=flag_gpu)
        #Segment image
        img_segmentation_02, flows, styles = model_trained_02.eval(img_original_Zo1, diameter=level_02_diam, channels= channels)
        del model_trained_02
        img_segmentation_02 = np.uint16(img_segmentation_02)        
        
        #Load model 03
        model_trained_03 = models.CellposeModel(pretrained_model=path_model_trained_level_03, gpu=flag_gpu)
        #Segment image
        img_segmentation_03, flows, styles = model_trained_03.eval(img_original_Zo1, diameter=level_03_diam, channels= channels)
        del model_trained_03
        img_segmentation_03 = np.uint16(img_segmentation_03)
        
        model_trained_04 = models.CellposeModel(pretrained_model=path_model_trained_level_04, gpu=flag_gpu)
        #Segment image
        img_segmentation_04, flows, styles = model_trained_04.eval(img_original_Zo1, diameter=level_04_diam, channels= channels)
        del model_trained_04
        img_segmentation_04 = np.uint16(img_segmentation_04)
        
        
        # Histograms of cell sizes
        cell_props_level_01 = get_props_per_cell(img_segmentation_01)
        cell_props_level_02 = get_props_per_cell(img_segmentation_02)
        cell_props_level_03 = get_props_per_cell(img_segmentation_03)
        cell_props_level_04 = get_props_per_cell(img_segmentation_04)
        
        count_C1, edges_C1 = np.histogram(get_areas(cell_props_level_01), bins=n_bins)
        count_C2, edges_C2 = np.histogram(get_areas(cell_props_level_02), bins=n_bins)
        count_C3, edges_C3 = np.histogram(get_areas(cell_props_level_03), bins=n_bins)
        count_C4, edges_C4 = np.histogram(get_areas(cell_props_level_04), bins=n_bins)
        
        
        #Plot with the max value of any bin of the nuclei segmentation
        max_value_bin = np.max([np.max(count_C1),np.max(count_C2),np.max(count_C3),np.max(count_C4)])

        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        
        plt.subplot(2, 2, 1)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C1[:-1]) / 10.)]
        plt.bar(labels_x,count_C1)
        plt.xlabel('Pxs x 10e1', fontsize=7)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 1')
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)

        plt.subplot(2, 2, 2)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C2[:-1]) / 10.)]
        plt.bar(labels_x,count_C2)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 2')
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        plt.xlabel('Pxs x 10e1', fontsize=7)

        plt.subplot(2, 2, 3)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C3[:-1]) / 10.)]
        plt.bar(labels_x,count_C3)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 3')
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        plt.xlabel('Pxs x 10e1', fontsize=7)

        plt.subplot(2, 2, 4)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C4[:-1]) / 10.)]
        plt.bar(labels_x,count_C4)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 4') 
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        plt.xlabel('Pxs x 10e1', fontsize=7)
        
        plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_area_cells_1.png'), dpi=400)
        
        plt.close('all')
        
        #Clean small cells
        _, good_cells_level_01_idx, noisy_cells_level_01_idx = detect_big_cells(cell_props_level_01, th_size=th_size_noisy_cells)
        _, good_cells_level_02_idx, noisy_cells_level_02_idx = detect_big_cells(cell_props_level_02, th_size=th_size_noisy_cells)
        _, good_cells_level_03_idx, noisy_cells_level_03_idx = detect_big_cells(cell_props_level_03, th_size=th_size_noisy_cells)
        _, good_cells_level_04_idx, noisy_cells_level_04_idx = detect_big_cells(cell_props_level_04, th_size=th_size_noisy_cells)
                
        # Clean segmentation
        for idx_01 in noisy_cells_level_01_idx:
            img_segmentation_01[img_segmentation_01==idx_01] = 0
        for idx_02 in noisy_cells_level_02_idx:
            img_segmentation_02[img_segmentation_02==idx_02] = 0
        for idx_03 in noisy_cells_level_03_idx:
            img_segmentation_03[img_segmentation_03==idx_03] = 0
        for idx_04 in noisy_cells_level_04_idx:
            img_segmentation_04[img_segmentation_04==idx_04] = 0
        
        del cell_props_level_01, cell_props_level_02, cell_props_level_03, cell_props_level_04, count_C1, edges_C1, count_C2, edges_C2, count_C3, edges_C3, count_C4, edges_C4
        
        # Histograms of cell sizes
        cell_props_level_01 = get_props_per_cell(img_segmentation_01)
        cell_props_level_02 = get_props_per_cell(img_segmentation_02)
        cell_props_level_03 = get_props_per_cell(img_segmentation_03)
        cell_props_level_04 = get_props_per_cell(img_segmentation_04)
        
        count_C1, edges_C1 = np.histogram(get_areas(cell_props_level_01), bins=n_bins)
        count_C2, edges_C2 = np.histogram(get_areas(cell_props_level_02), bins=n_bins)
        count_C3, edges_C3 = np.histogram(get_areas(cell_props_level_03), bins=n_bins)
        count_C4, edges_C4 = np.histogram(get_areas(cell_props_level_04), bins=n_bins)
        
        #Plot with the max value of any bin of the nuclei segmentation
        max_value_bin = np.max([np.max(count_C1),np.max(count_C2),np.max(count_C3),np.max(count_C4)])
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        
        plt.subplot(2, 2, 1)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C1[:-1]) / 10.)]
        plt.bar(labels_x,count_C1)
        plt.xlabel('Pxs x 10e1', fontsize=7)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 1')
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)

        plt.subplot(2, 2, 2)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C2[:-1]) / 10.)]
        plt.bar(labels_x,count_C2)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 2')
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        plt.xlabel('Pxs x 10e1', fontsize=7)

        plt.subplot(2, 2, 3)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C3[:-1]) / 10.)]
        plt.bar(labels_x,count_C3)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 3')
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        plt.xlabel('Pxs x 10e1', fontsize=7)

        plt.subplot(2, 2, 4)
        labels_x = [str(x) for x in np.uint32(np.array(edges_C4[:-1]) / 10.)]
        plt.bar(labels_x,count_C4)
        plt.ylim(0, max_value_bin)
        plt.gca().set_title('Area level 4') 
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        plt.xlabel('Pxs x 10e1', fontsize=7)
        
        plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_area_cells_2.png'), dpi=400)
        
        model_trained_nuclei = models.CellposeModel(model_type='nuclei', gpu=flag_gpu)
        img_segmentation_nuclei, flows, styles = model_trained_nuclei.eval(img_original_nuclei, diameter=diam_nuclei, channels= channels)
        
        n_cells_level_01 = len(np.unique(img_segmentation_01)) -1 # to not count the label background
        n_cells_level_02 = len(np.unique(img_segmentation_02)) -1
        n_cells_level_03 = len(np.unique(img_segmentation_03)) -1
        n_cells_level_04 = len(np.unique(img_segmentation_04)) -1
        
        #Compute distinctive cells
        n_cells_level_02_added = n_cells_level_02 - n_cells_level_01
        n_cells_level_03_added = n_cells_level_03 - n_cells_level_02
        n_cells_level_04_added = n_cells_level_04 - n_cells_level_03
        
        
        img_segmentation_01_expanded = expand_labels(img_segmentation_01, distance=10)
        img_segmentation_02_expanded = expand_labels(img_segmentation_02, distance=10)
        img_segmentation_03_expanded = expand_labels(img_segmentation_03, distance=10)
        img_segmentation_04_expanded = expand_labels(img_segmentation_04, distance=10)
        
        
        edge_01 = filters.sobel(img_segmentation_01_expanded)
        edge_02 = filters.sobel(img_segmentation_02_expanded)
        edge_03 = filters.sobel(img_segmentation_03_expanded)
        edge_04 = filters.sobel(img_segmentation_04_expanded)
        
        edge_01 = edge_01 > 1.e-10
        edge_02 = edge_02 > 1.e-10
        edge_03 = edge_03 > 1.e-10
        edge_04 = edge_04 > 1.e-10
        
        edge_01 = dilation(edge_01, footprint=square(10))
        edge_02 = dilation(edge_02, footprint=square(10))
        edge_03 = dilation(edge_03, footprint=square(10))
        edge_04 = dilation(edge_04, footprint=square(10))
        
        sum_edges = np.uint8(edge_01) + np.uint8(edge_02) + np.uint8(edge_03) + np.uint8(edge_04)
        
        #Matching nuclei and cells in different levels
        
        _, _, matching_pairs_a_to_b_non_zero_level_01 = matching_label_pairs(img_segmentation_nuclei, img_segmentation_01_expanded, min_pixels=min_pixels_matching)
        _, _, matching_pairs_a_to_b_non_zero_level_02 = matching_label_pairs(img_segmentation_nuclei, img_segmentation_02_expanded, min_pixels=min_pixels_matching)
        _, _, matching_pairs_a_to_b_non_zero_level_03 = matching_label_pairs(img_segmentation_nuclei, img_segmentation_03_expanded, min_pixels=min_pixels_matching)
        _, _, matching_pairs_a_to_b_non_zero_level_04 = matching_label_pairs(img_segmentation_nuclei, img_segmentation_04_expanded, min_pixels=min_pixels_matching)
        
        # Level 1
        img_segmentation_membrane_multiple_nuclei_level_01, img_segmentation_nuclei_in_same_cells_level_01, _ =\
            get_correspondance_segmentations(img_segmentation_01_expanded, img_segmentation_nuclei, min_pixels_matching=min_pixels_matching)
        
        # Level 2
        img_segmentation_membrane_multiple_nuclei_level_02, img_segmentation_nuclei_in_same_cells_level_02, _ =\
            get_correspondance_segmentations(img_segmentation_02_expanded, img_segmentation_nuclei, min_pixels_matching=min_pixels_matching)
        
        # Level 3
        img_segmentation_membrane_multiple_nuclei_level_03, img_segmentation_nuclei_in_same_cells_level_03, _ =\
            get_correspondance_segmentations(img_segmentation_03_expanded, img_segmentation_nuclei, min_pixels_matching=min_pixels_matching)
            
        # Level 4
        img_segmentation_membrane_multiple_nuclei_level_04, img_segmentation_nuclei_in_same_cells_level_04, _ =\
            get_correspondance_segmentations(img_segmentation_04_expanded, img_segmentation_nuclei, min_pixels_matching=min_pixels_matching)    
        
        #Image composition
        img_composition = (img_original_nuclei / 2) + (img_original_Zo1 / 2)
        
        filename_composition = os.path.join(folder_output, sample_name + '_composite_nuclei_Zo1.png')
        img_composition_bgr = cv2.cvtColor((img_composition * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_composition, img_composition_bgr)
        
        ##---- Summary image ----
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
        plt.subplot(2, 2, 1)
        plt.imshow(img_original_Zo1,cmap='gray')
        plt.gca().set_title('Zo1', fontsize=fontsize)
        plt.subplot(2, 2, 2)
        plt.imshow(sum_edges)
        plt.gca().set_title('Edges (yellow:more defined)', fontsize=fontsize)
        plt.subplot(2, 2, 3)
        plt.imshow(img_original_nuclei,cmap='gray')
        plt.gca().set_title('Nuclei', fontsize=fontsize)
        plt.subplot(2, 2, 4)
        plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
        plt.gca().set_title('Nuclei seg', fontsize=fontsize)
        
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(5, 3, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
        plt.subplot(5, 3, 1)
        plt.imshow(img_composition,cmap='gray')
        plt.gca().set_title('Composition', fontsize=fontsize)
        
        plt.subplot(5, 3, 2)
        plt.imshow(sum_edges)
        plt.gca().set_title('Edges (yellow:more defined)', fontsize=fontsize)
        
        plt.subplot(5, 3, 3)
        plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
        plt.gca().set_title('Nuclei seg', fontsize=fontsize)
        
        plt.subplot(5, 3, 4)
        plt.imshow(img_segmentation_01)
        plt.gca().set_title('Segmentation 01', fontsize=fontsize)
        
        plt.subplot(5, 3, 5)
        plt.imshow(img_segmentation_membrane_multiple_nuclei_level_01,cmap='gist_ncar')
        plt.gca().set_title('Membranes multiple nuclei level 1', fontsize=fontsize)
        
        plt.subplot(5, 3, 6)
        plt.imshow(img_segmentation_nuclei_in_same_cells_level_01,cmap='gist_ncar')
        plt.gca().set_title('Nuclei in same cells level 1', fontsize=fontsize)
        
        
        plt.subplot(5, 3, 7)
        plt.imshow(img_segmentation_02)
        plt.gca().set_title('Segmentation 02', fontsize=fontsize)
        
        plt.subplot(5, 3, 8)
        plt.imshow(img_segmentation_membrane_multiple_nuclei_level_02,cmap='gist_ncar')
        plt.gca().set_title('Membranes multiple nuclei level 2', fontsize=fontsize)
        
        plt.subplot(5, 3, 9)
        plt.imshow(img_segmentation_nuclei_in_same_cells_level_02,cmap='gist_ncar')
        plt.gca().set_title('Nuclei in same cells level 2', fontsize=fontsize)
        
        
        plt.subplot(5, 3, 10)
        plt.imshow(img_segmentation_03)
        plt.gca().set_title('Segmentation 03', fontsize=fontsize)
        
        plt.subplot(5, 3, 11)
        plt.imshow(img_segmentation_membrane_multiple_nuclei_level_03,cmap='gist_ncar')
        plt.gca().set_title('Membranes multiple nuclei level 3', fontsize=fontsize)
        
        plt.subplot(5, 3, 12)
        plt.imshow(img_segmentation_nuclei_in_same_cells_level_03,cmap='gist_ncar')
        plt.gca().set_title('Nuclei in same cells level 3', fontsize=fontsize)
        
        
        plt.subplot(5, 3, 13)
        plt.imshow(img_segmentation_04)
        plt.gca().set_title('Segmentation 04', fontsize=fontsize)
        
        plt.subplot(5, 3, 14)
        plt.imshow(img_segmentation_membrane_multiple_nuclei_level_04,cmap='gist_ncar')
        plt.gca().set_title('Membranes multiple nuclei level 4', fontsize=fontsize)
        
        plt.subplot(5, 3, 15)
        plt.imshow(img_segmentation_nuclei_in_same_cells_level_04,cmap='gist_ncar')
        plt.gca().set_title('Nuclei in same cells level 4', fontsize=fontsize)
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # plt.show()
        
        plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_sharpness_cells.png'), dpi=400)
        
        
        ########### Background detection #################
        structure_closing= disk(disk_diameter)
        
        cells_and_nuclei_mask = (img_segmentation_01_expanded > 0) | (img_segmentation_02_expanded > 0) | (img_segmentation_03_expanded > 0) | (img_segmentation_nuclei > 0) | (img_segmentation_04_expanded > 0)
        largest_component_background, labeled_array_background, mask_closed = get_large_empty_spaces(cells_and_nuclei_mask, structure_closing)
        
        ########## no_cell area candidates ############
        
        #Here it is to include or not level 04
        cells_mask = (img_segmentation_01_expanded > 0) | (img_segmentation_02_expanded > 0) | (img_segmentation_03_expanded > 0) | largest_component_background | (img_segmentation_04_expanded > 0)
        cells_mask_closed = ndimage.binary_closing(cells_mask, structure=structure_closing, border_value=1)
        
        largest_component_no_cell, labeled_array_no_cell, _ = get_large_empty_spaces(cells_mask_closed, structure_closing)
        
        ######### Nuclei in no_cell zone #####
        
        _, _, matching_pairs_a_to_b_non_zero = matching_label_pairs(largest_component_no_cell, img_segmentation_nuclei, min_pixels=min_pixels_matching)
        
        list_nuclei_in_syncitial = [pair[1] for pair in matching_pairs_a_to_b_non_zero]
        img_segmentation_nuclei_in_syncitial = get_img_from_idx_cells(img_segmentation_nuclei, list_nuclei_in_syncitial)
        
        # Detect edges of the positive areas
        # Using Sobel edge detection
        edges_no_cell = filters.sobel(largest_component_no_cell.astype(float))
        edges_no_cell = edges_no_cell > 0.00001
        
        edges_background = filters.sobel(largest_component_background.astype(float))
        edges_background = edges_background > 0.00001
        # Stats
        
        # Count the number of positive (True) pixels
        px_syncitial = np.sum(largest_component_no_cell)
        print("Number of no_cell pixels:", px_syncitial)
        
        plate = np.logical_not(largest_component_background)
        px_plate = np.sum(plate)
        print("Number of plate pixels:", px_plate)
        
        ratio_syncitial_plate = float(px_syncitial) / float(px_plate + 0.0000001)
        #print("ratio_syncitial_plate:", ratio_syncitial_plate)
        
        total_nuclei = len(np.unique(img_segmentation_nuclei))
        nuclei_in_syncitial = len(list_nuclei_in_syncitial)
        ratio_nuclei_syncitial = float(nuclei_in_syncitial) / float(total_nuclei + 0.0000001)
        print("total_nuclei:", total_nuclei)
        #print("nuclei_in_syncitial:", nuclei_in_syncitial)
        #print("ratio_nuclei_syncitial:", ratio_nuclei_syncitial)
                
        n_cells_multiple_nuclei_level_01 = len(np.unique(img_segmentation_membrane_multiple_nuclei_level_01))-1 #0 is background
        n_cells_multiple_nuclei_level_02 = len(np.unique(img_segmentation_membrane_multiple_nuclei_level_02))-1 #0 is background
        n_cells_multiple_nuclei_level_03 = len(np.unique(img_segmentation_membrane_multiple_nuclei_level_03))-1 #0 is background
        n_cells_multiple_nuclei_level_04 = len(np.unique(img_segmentation_membrane_multiple_nuclei_level_04))-1 #0 is background
        
        n_nuclei_sharing_membrane_level_01 = len(np.unique(img_segmentation_nuclei_in_same_cells_level_01))-1 #0 is background
        n_nuclei_sharing_membrane_level_01 = 0 if n_cells_multiple_nuclei_level_01 == 0 else n_nuclei_sharing_membrane_level_01 #if no multinuclei cells, no nuclei sharing cells
        n_nuclei_sharing_membrane_level_02 = len(np.unique(img_segmentation_nuclei_in_same_cells_level_02))-1 #0 is background
        n_nuclei_sharing_membrane_level_02 = 0 if n_cells_multiple_nuclei_level_02 == 0 else n_nuclei_sharing_membrane_level_02 #if no multinuclei cells, no nuclei sharing cells
        n_nuclei_sharing_membrane_level_03 = len(np.unique(img_segmentation_nuclei_in_same_cells_level_03))-1 #0 is background
        n_nuclei_sharing_membrane_level_03 = 0 if n_cells_multiple_nuclei_level_03 == 0 else n_nuclei_sharing_membrane_level_03 #if no multinuclei cells, no nuclei sharing cells
        n_nuclei_sharing_membrane_level_04 = len(np.unique(img_segmentation_nuclei_in_same_cells_level_04))-1 #0 is background
        n_nuclei_sharing_membrane_level_04 = 0 if n_cells_multiple_nuclei_level_04 == 0 else n_nuclei_sharing_membrane_level_04 #if no multinuclei cells, no nuclei sharing cells
        
        total_nuclei_in_multinuclei_cells = n_nuclei_sharing_membrane_level_04 + nuclei_in_syncitial
        ratio_nuclei_in_multinuclei_cells = total_nuclei_in_multinuclei_cells / (total_nuclei + 0.0000001)
        
        # Normalization to compute the slope
        n_cells_level_01_norm = n_cells_level_01 / n_cells_level_04
        n_cells_level_02_norm = n_cells_level_02 / n_cells_level_04
        n_cells_level_03_norm = n_cells_level_03 / n_cells_level_04
        n_cells_level_04_norm = n_cells_level_04 / n_cells_level_04
        
        # Fit linear regression model
        vector_n_cells = np.array([n_cells_level_01_norm, n_cells_level_02_norm, n_cells_level_03_norm, n_cells_level_04_norm])
        levels_cells = np.array([1, 2, 3, 4])
        linear_results = stats.linregress(levels_cells, vector_n_cells)
        plt.figure()
        plt.plot(levels_cells, vector_n_cells, 'o', label='n cells')
        plt.plot(levels_cells, linear_results.intercept + linear_results.slope*levels_cells, 'r', label='fitted line')
        plt.xlabel('Zo1 strength')
        plt.ylabel('n cells norm')
        plt.ylim(0, 1.1)
        plt.xlim(0,max(levels_cells)+1)        
        plt.legend()
        plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_n_cells_fitting.png'), dpi=400)
        
        txt_output = os.path.join(folder_output, sample_name + '_summary.txt')
        f = open(txt_output, "w")
        #f.write("Now the file has more content!")
        f.write(sample_name + '\n')
                
        f.write('------------------------------------------------------- \n')
        f.write('Unique cells per level:\n')
        f.write('Level_04: ' + str( n_cells_level_04_added) + '\n')
        f.write('Level_03: ' + str( n_cells_level_03_added) + '\n')
        f.write('Level_02: ' + str( n_cells_level_02_added) + '\n')
        f.write('Level_01: ' + str( n_cells_level_01) + '\n')
                
        f.write('------------------------------------------------------- \n')
        f.write('Total cells per level:\n')
        f.write('Level_04: ' + str( n_cells_level_04) + '\n')
        f.write('Level_03: ' + str( n_cells_level_03) + '\n')
        f.write('Level_02: ' + str( n_cells_level_02) + '\n')
        f.write('Level_01: ' + str( n_cells_level_01) + '\n')
        f.write('\n')
        f.write('Normalization:\n')
        f.write('Level_04: ' + str( n_cells_level_04_norm) + '\n')
        f.write('Level_03: ' + str( n_cells_level_03_norm) + '\n')
        f.write('Level_02: ' + str( n_cells_level_02_norm) + '\n')
        f.write('Level_01: ' + str( n_cells_level_01_norm) + '\n')
        f.write('\n')
        f.write('Linear fitting:\n')
        f.write('slope: ' + str( linear_results.slope) + '\n')
        f.write('r_value: ' + str( linear_results.rvalue) + '\n')
        f.write('std_err: ' + str( linear_results.stderr) + '\n')
        
        f.write('------------------------------------------------------- \n')
        f.write('Multi-nuclei cells per level:\n')
        f.write('\n')
        f.write('Level 01: ' + str( n_cells_multiple_nuclei_level_01) + '\n')
        f.write('Nuclei sharing cells level 01: ' + str( n_nuclei_sharing_membrane_level_01) + '\n')
        f.write('\n')
        f.write('Level 02: ' + str( n_cells_multiple_nuclei_level_02) + '\n')
        f.write('Nuclei sharing cells level 02: ' + str( n_nuclei_sharing_membrane_level_02) + '\n')
        f.write('\n')
        f.write('Level 03: ' + str( n_cells_multiple_nuclei_level_03) + '\n')
        f.write('Nuclei sharing cells level 03: ' + str( n_nuclei_sharing_membrane_level_03) + '\n')
        f.write('\n')
        f.write('Level 04: ' + str( n_cells_multiple_nuclei_level_04) + '\n')
        f.write('Nuclei sharing cells level 04: ' + str( n_nuclei_sharing_membrane_level_04) + '\n')
        
        f.write('------------------------------------------------------- \n')
        f.write('Total nuclei: ' + str( total_nuclei) + '\n')
        f.write('Nuclei free: ' + str( nuclei_in_syncitial) + '\n')
        f.write('Ratio nuclei free: ' + str( ratio_nuclei_syncitial) + '\n')
        f.write('\n')
        f.write('Number of pixels in cell free area: ' + str(px_syncitial) + '\n')
        f.write('Number of pixels in plate (after removing background): ' + str( px_plate) + '\n')
        f.write('Ratio free area - pxs in plate: ' + str( ratio_syncitial_plate) + '\n')
        f.write('------------------------------------------------------- \n')
        f.write('Total nuclei: ' + str( total_nuclei) + '\n')
        f.write('Total nuclei in multinuclei cells (at level 4) or free: ' + str( total_nuclei_in_multinuclei_cells) + '\n')
        f.write('Ratio nuclei in multinuclei cells or no cells vs total nuclei: ' + str( ratio_nuclei_in_multinuclei_cells) + '\n')
        
        f.close()
        
        #Output
        img_segmentation_nuclei_in_syncitial = draw_roi_over_image(img_original_nuclei, img_segmentation_nuclei_in_syncitial)
        img_composition_syncitial_edge = draw_mask_over_image(img_composition, edges_no_cell, color_mask = [255, 0, 0])
        img_composition_syncitial_edge = draw_mask_over_image_rgb(img_composition_syncitial_edge, edges_background, color_mask = [0, 40, 40])
        
        filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_freeNucleiR_backC.png')
        img_composition_syncitial_edge_bgr = cv2.cvtColor(img_composition_syncitial_edge, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_syncitial_back, img_composition_syncitial_edge_bgr)
        
        #largest_component_no_cell
        img_composition_syncitial = overlap_mask_over_image_rgb(img_composition_syncitial_edge, largest_component_no_cell, color_add = color_add_free_nuclei) # [50, 0, 0]
        filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_freeNucleiR_backC_fill.png')
        img_composition_syncitial_edge_bgr = cv2.cvtColor(img_composition_syncitial, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_syncitial_back, img_composition_syncitial_edge_bgr)
        
        #largest_component_no_cell
        img_composition_syncitial = overlap_mask_over_image_rgb(img_composition_syncitial, largest_component_background, color_add = color_add_back) # [0, 40, 40]
        filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_freeNucleiR_backC_fill_2.png')
        img_composition_syncitial_edge_bgr = cv2.cvtColor(img_composition_syncitial, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_syncitial_back, img_composition_syncitial_edge_bgr)
        
        # Draw edges cells
        # For a correct overlapping, from the weakest edge to the stronger ones
        
        img_composition_edges = draw_mask_over_image(img_original_Zo1, edge_01, color_mask = color_edges_01) # [255, 17, 0]
        filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_level01.png')
        img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
        
        img_composition_edges = draw_mask_over_image(img_original_Zo1, edge_02, color_mask = color_edges_02) # [255, 250, 0]
        filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_level02.png')
        img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
        
        img_composition_edges = draw_mask_over_image(img_original_Zo1, edge_03, color_mask = color_edges_03) # [21, 250, 0]
        filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_level03.png')
        img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
        
        # Level 04 in the composition to see multi-nuclei cells
        img_composition_level04 = draw_mask_over_image(img_composition, edge_04, color_mask = color_edges_04) # [0, 40, 255]
        filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_level04_in_composition.png')
        img_composition_edges_bgr = cv2.cvtColor(img_composition_level04, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
        
                
        # Composition of images
        
        img_composition_edges = draw_mask_over_image(img_original_Zo1, edge_04, color_mask = color_rainbow_sum_1) # [0, 40, 255]
        filename_edges_bgr = os.path.join(folder_output, sample_name + '_edges_level04.png')
        img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
        
        img_composition_edges = draw_mask_over_image_rgb(img_composition_edges, edge_03, color_mask = color_rainbow_sum_2) # [21, 250, 0]
        img_composition_edges = draw_mask_over_image_rgb(img_composition_edges, edge_02, color_mask = color_rainbow_sum_3) # [255, 250, 0]
        img_composition_edges = draw_mask_over_image_rgb(img_composition_edges, edge_01, color_mask = color_rainbow_sum_4) # [255, 17, 0]
        
        filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_rainbow.png')
        img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
        
        
        
        sum_cell_seg = np.uint8(img_segmentation_01_expanded > 0) +  np.uint8(img_segmentation_02_expanded > 0) + np.uint8(img_segmentation_03_expanded > 0) + np.uint8(img_segmentation_04_expanded > 0)
        img_composition_sum_cell = draw_mask_over_image(img_composition,                sum_cell_seg == 1, color_mask = color_rainbow_sum_1) # [0, 40, 255]
        img_composition_sum_cell = draw_mask_over_image_rgb(img_composition_sum_cell,   sum_cell_seg == 2, color_mask = color_rainbow_sum_2) # [21, 250, 0]
        img_composition_sum_cell = draw_mask_over_image_rgb(img_composition_sum_cell,   sum_cell_seg == 3, color_mask = color_rainbow_sum_3) # [255, 250, 0]
        img_composition_sum_cell = draw_mask_over_image_rgb(img_composition_sum_cell,   sum_cell_seg == 4, color_mask = color_rainbow_sum_4) # [255, 17, 0]
        
        filename_sum_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_rainbow.png')
        img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
        
        #img_composition_sum_cell = draw_mask_over_image(img_composition, edge_04, color_mask = [0, 0, 100])
        img_composition_sum_cell = (img_composition * 255).astype(np.uint8)
        img_composition_sum_cell = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_GRAY2BGR)
        img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, sum_cell_seg == 1, color_add = color_add_rainbow_1) # [0, 0, 80]
        img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, sum_cell_seg == 2, color_add = color_add_rainbow_2) # [0, 80, 0]
        img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, sum_cell_seg == 3, color_add = color_add_rainbow_3) # [40, 40, 0]
        img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, sum_cell_seg == 4, color_add = color_add_rainbow_4) # [80, 0, 0]
        filename_sum_bgr = os.path.join(folder_output, sample_name + '_rainbow_watermark.png')
        img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
        
        img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, largest_component_background>0, color_add = color_add_back) # [0, 40, 40]
        filename_sum_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_rainbow_back_watermark.png')
        img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
        
        
        img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, largest_component_no_cell>0, color_add = [80, 0, 80])
        filename_sum_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_rainbow_back_syncitial_watermark.png')
        img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
        
        
        #Draw nuclei in composition
        img_composition_nuclei = draw_mask_over_image(img_composition, img_segmentation_nuclei > 0, color_mask = color_nuclei) # [0, 150, 150]
        filename_nuclei_bgr = os.path.join(folder_output, sample_name + '_segLevel04B.png')
        img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_nuclei_bgr, img_composition_nuclei_bgr)
        
        img_composition_nuclei = draw_mask_over_image_rgb(img_composition_nuclei, edge_04, color_mask = color_edges_04)
        filename_nuclei_bgr = os.path.join(folder_output, sample_name + '_nucleiC_segLevel04B.png')
        img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_nuclei_bgr, img_composition_nuclei_bgr)
        
        img_composition_nuclei = overlap_mask_over_image_rgb(img_composition_nuclei, largest_component_no_cell>0, color_add = color_add_free_nuclei) # [100, 10, 10]
        filename_nuclei_bgr = os.path.join(folder_output, sample_name + '_nucleiC_segLevel04B_freenucleiR.png')
        img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_nuclei_bgr, img_composition_nuclei_bgr)
        
        img_composition_nuclei = overlap_mask_over_image_rgb(img_composition_nuclei, img_segmentation_membrane_multiple_nuclei_level_04>0, color_add = color_add_multiple_nuclei) # [80, 80, 10]
        filename_nuclei_bgr = os.path.join(folder_output, sample_name + '_nucleiC_segLevel04B_freenucleiR_multinucleiY.png')
        img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_nuclei_bgr, img_composition_nuclei_bgr)
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(4, 3, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
        
        plt.subplot(4, 3, 1)
        plt.imshow(cells_and_nuclei_mask,cmap='gist_ncar')
        plt.gca().set_title('cells_and_nuclei_mask', fontsize=fontsize)
        
        plt.subplot(4, 3, 2)
        plt.imshow(largest_component_background,cmap='gist_ncar')
        plt.gca().set_title('largest_component_background', fontsize=fontsize)
        
        plt.subplot(4, 3, 4)
        plt.imshow(cells_mask,cmap='gist_ncar')
        plt.gca().set_title('cells_mask', fontsize=fontsize)
        
        plt.subplot(4, 3, 5)
        plt.imshow(cells_mask_closed,cmap='gist_ncar')
        plt.gca().set_title('cells_mask_closed', fontsize=fontsize)
        
        
        plt.subplot(4, 3, 7)
        plt.imshow(largest_component_no_cell,cmap='gist_ncar')
        plt.gca().set_title('largest_component_no_cell', fontsize=fontsize)
        
        plt.subplot(4, 3, 8)
        plt.imshow(img_segmentation_nuclei_in_syncitial,cmap='gist_ncar')
        plt.gca().set_title('img_segmentation_nuclei_free', fontsize=fontsize)
        
        plt.subplot(4, 3, 9)
        plt.imshow(edges_no_cell,cmap='gist_ncar')
        plt.gca().set_title('edges_no_cell', fontsize=fontsize)
        
        plt.subplot(4, 3, 10)
        plt.imshow(img_segmentation_nuclei_in_syncitial)
        plt.gca().set_title('nuclei_free', fontsize=fontsize)
        
        plt.subplot(4, 3, 11)
        plt.imshow(img_composition_syncitial_edge)
        plt.gca().set_title('img_composition_free_edge', fontsize=fontsize)
                
        plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_debug.png'), dpi=400)
          
        #Final summary figure
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(3, 3, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
        plt.subplot(3, 3, 1)
        plt.imshow(img_original_Zo1,cmap='gray')
        plt.gca().set_title('Zo1', fontsize=fontsize)
        plt.subplot(3, 3, 2)
        plt.imshow(sum_edges)
        plt.gca().set_title('Edges (yellow:more defined)', fontsize=fontsize)
        plt.subplot(3, 3, 4)
        plt.imshow(img_original_nuclei,cmap='gray')
        plt.gca().set_title('Nuclei', fontsize=fontsize)
        plt.subplot(3, 3, 5)
        plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
        plt.gca().set_title('Nuclei seg', fontsize=fontsize)
        
        plt.subplot(3, 3, 7)
        plt.imshow(img_composition,cmap='gray')
        plt.gca().set_title('Composition', fontsize=fontsize)
        
        plt.subplot(3, 3, 8)
        plt.imshow(img_composition_syncitial_edge)
        plt.gca().set_title('Free: ' + str(int(ratio_syncitial_plate * 100)) + '%', fontsize=fontsize)
        
        plt.subplot(3, 3, 9)
        plt.imshow(img_segmentation_nuclei_in_syncitial)
        plt.gca().set_title('Nuclei in free zone: ' + str(int(ratio_nuclei_syncitial * 100)) + '%', fontsize=fontsize)
        
        plt.savefig(os.path.join(folder_output, sample_name + '_summary.png'), dpi=400)
        
        plt.close('all')
        
        print('------------------------------------')
 
