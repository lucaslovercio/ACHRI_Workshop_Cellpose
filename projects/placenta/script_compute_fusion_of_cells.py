#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:22:46 2024

@author: lucas
"""

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

###################################   PARAMETERS   #########################

pattern_channel_dapi = 'DAPI_'
pattern_channel_Zo1 = 'ZO1_'
pattern_channel_mask = 'MASK_'
pattern_channel_cyto = ''

folder_images = '' # Folder where all your images must be

folder_models = '' # The two next models must be in the same folder
name_model_level_01 = 'well_defined_diam_100_cyto_100_ji_0.4027.114269'
name_model_level_02 = 'less_defined_model_nuclei_diam_50_ji_0.405.265445'

path_model_nuclei_trained  = 'DAPI_diam50_cyto2.792364' # Complete the full path for this model

flag_normalize = True # Recommended: True
flag_gpu = False # True is you have a configured GPU for PyTorch
flag_has_empty_space = False # Does your image have a space without cells?
flag_cyto = False # Is there a cyto marker to help to detect the empty space in your image?
flag_remove_objects_in_edge = True # Do you want to remove from the analysis incomplete cells and nuclei in the edge of the image?

###################################  MORE PARAMETERS   #########################

channels = [[0,0]] #Same channels as training

# Parameters for 20x
diam_nuclei = 100
margin = 10
distance_expand_labels = 5
min_pixels_matching = 1000 # Ife data #2000

th_size_noisy_cells = 1000
disk_diameter_background = 90 #90
disk_diameter_free_cell = 30
th_size_no_cell_space = 40000 # only used when flag_cyto=True
fontsize = 10

level_01_diam = None # 100
level_04_diam = None # 100

n_bins = 10

# Colours of outputs

color_add_free_nuclei = [50, 0, 0]
color_add_back = [0, 40, 40]

color_edges_01 = [255, 17, 0]
color_edges_04 = [0, 40, 255]

color_rainbow_sum_1 = [21, 250, 0]
color_rainbow_sum_4 = [255, 17, 0]

color_add_rainbow_1 = [0, 80, 0]
color_add_rainbow_4 = [0, 0, 80]

color_nuclei = [0, 150, 150]
color_add_multiple_nuclei = [80, 80, 10]

color_excluded_region = [150, 0, 150]
color_plate = [150, 150, 0]


##############################################################################
##############################################################################
##############################################################################
##############################################################################

import numpy as np
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
from aux_functions.functionPercNorm import functionPercNorm, get_one_channel
from skimage.segmentation import expand_labels
from skimage.morphology import dilation, square
from skimage import filters
from quantify_segmentation import matching_label_pairs, matching_label_pairs_perc, get_correspondance_segmentations_perc, get_img_from_idx_cells, \
    get_props_per_cell, get_areas, detect_big_cells, remove_overlapping_segmentation, get_large_empty_spaces, remove_objects_in_edge
from scipy import ndimage
# from scipy.ndimage import label #, gaussian_filter
from skimage.morphology import disk
from skimage.filters import median
from aux_functions.draw_roi_over_image import draw_roi_over_image, draw_mask_over_image, draw_mask_over_image_rgb, overlap_mask_over_image_rgb
from aux_functions.filepath_management import get_sample_name, get_image_filenames
import cv2
from scipy import stats
import pandas as pd


def correct_upper_level_segmentation(img_segmentation_lower, img_segmentation_upper, min_pixels):
    # If a cell is missing in level less-defined, add it
    matching_pairs, matching_pairs_non_zero_left, matching_pairs_a_to_b_non_zero_level_01_04 = \
        matching_label_pairs(img_segmentation_lower, img_segmentation_upper, min_pixels=min_pixels)
        
    max_label = np.max(img_segmentation_upper) + 1
    
    for pair in matching_pairs_non_zero_left:
        if pair[1] == 0: # No correspondance to the right
            #copy cell in level 1 to level 2
            img_segmentation_upper[img_segmentation_lower==pair[0]] = max_label
            max_label = max_label + 1
            
    return img_segmentation_upper


def main():
    
    path_model_trained_level_01  = os.path.join(folder_models, name_model_level_01)
    path_model_trained_level_04  = os.path.join(folder_models, name_model_level_02)
    
    image_filenames = get_image_filenames(folder_images, substring_pattern = pattern_channel_dapi)
    # image_filenames = image_filenames[0:4]
    
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=["SampleName", "FusionIndex([B+C]/A)", "RatioFirmCellsVsUniCells(E/D)", "RatioPorousCellsVsUniCells([F-E]/D)", \
                               "RatioNucleiInMultinucleateCellsVsTotalNuclei(C/A)", \
                               "", "RatioFirmCellsVsTotalCells(E/F)", "RatioNucleiInUnicellsVsTotalNuclei(D/A)"]) # Leave empty column
    
    vector_FusionIndex_wt = []
    vector_FusionIndex_mut = []
    vector_FusionIndex_het = []
    
    vector_RatioFirmCellsVsUniCells_wt = [] 
    vector_RatioFirmCellsVsUniCells_mut = []
    vector_RatioFirmCellsVsUniCells_het = []
    
    vector_RatioPorousCellsVsUniCells_wt = [] 
    vector_RatioPorousCellsVsUniCells_mut = []
    vector_RatioPorousCellsVsUniCells_het = []
    
    vector_RatioFirmCellsVsTotalCells_wt = []
    vector_RatioFirmCellsVsTotalCells_mut = []
    vector_RatioFirmCellsVsTotalCells_het = []
    
    vector_RatioNucleiInUnicellsVsTotalNuclei_wt = []
    vector_RatioNucleiInUnicellsVsTotalNuclei_mut = []
    vector_RatioNucleiInUnicellsVsTotalNuclei_het = []
    
    vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_wt = []
    vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_mut = []
    vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_het = []
    
    for filename_nuclei in image_filenames:
        plt.close('all')
        print('------------------------------------')
        print('Reading: ' + filename_nuclei)    
        
        sample_name = get_sample_name(filename_nuclei)
        
        filename_Zo1 = filename_nuclei.replace(pattern_channel_dapi, pattern_channel_Zo1)
        filename_cyto = filename_nuclei.replace(pattern_channel_dapi, pattern_channel_cyto)
        filename_mask = filename_nuclei.replace(pattern_channel_dapi, pattern_channel_mask)
        
        image_input_nuclei    = os.path.join(folder_images,filename_nuclei)
        image_input_Zo1  = os.path.join(folder_images,filename_Zo1)
        image_input_cyto  = os.path.join(folder_images,filename_cyto)
        image_input_mask  = os.path.join(folder_images,filename_mask)
        
        
        #Load image (first channel)
        img_original_Zo1 = imread(image_input_Zo1)
        img_original_nuclei = imread(image_input_nuclei)
        
        flag_masked = os.path.exists(image_input_mask)
        
        flag_wt = 'wt' in sample_name.lower()
        flag_het = 'het' in sample_name.lower()
        flag_mut = ('ko' in sample_name.lower()) or ('mut' in sample_name.lower())
        
        str_margin = '_Margin' + str(margin) + '_keeping_nuclei_dapiShuhibaTrained_percMatching' + \
            '_SEBack' + str(disk_diameter_background) + '_AndNoCellAssoc' + str(disk_diameter_free_cell)
        
        str_remove_objects_in_edge = ''
        if flag_remove_objects_in_edge:
            str_remove_objects_in_edge = '_ObjEdgeRemoved'
            
        str_cyto_marker = '_NoCytoMarker'
        if flag_cyto:
            str_cyto_marker = '_WithCytoMarker'
            
        str_has_empty_space = '_NoEmptySpace'
        if flag_has_empty_space:
            str_has_empty_space = '_EmptySpace'
            
        str_masked_area = '_NoMaskedArea'
        if flag_masked:
            str_masked_area = '_WithMaskedArea'
        
        
        folder_output = os.path.join(folder_images, sample_name + str_has_empty_space + str_cyto_marker + str_remove_objects_in_edge + str_margin + str_masked_area)
        folder_output_intermediate_output = os.path.join(folder_output, 'intermediate_output')
        
        
        if flag_masked:
            img_original_mask = imread(image_input_mask)
            stacked = np.stack([img_original_mask[:,:,0], img_original_mask[:,:,1], img_original_mask[:,:,2]], axis=0)
            mask = np.std(stacked, axis=0, ddof=0) > 0.0001
            
        if flag_cyto:
            img_original_cyto = imread(image_input_cyto)
        
        
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
                if flag_cyto:
                    img_original_cyto = get_one_channel(img_original_cyto)
                    img_original_cyto = functionPercNorm( np.single(img_original_cyto))
            
            
            model_trained_01 = models.CellposeModel(pretrained_model=path_model_trained_level_01, gpu=flag_gpu)
            img_segmentation_01, _, _ = model_trained_01.eval(img_original_Zo1, diameter=level_01_diam, channels= channels)
            del model_trained_01
            img_segmentation_01 = np.uint16(img_segmentation_01)
            
            model_trained_04 = models.CellposeModel(pretrained_model=path_model_trained_level_04, gpu=flag_gpu)
            img_segmentation_04, _, _ = model_trained_04.eval(img_original_Zo1, diameter=level_04_diam, channels= channels)
            del model_trained_04
            img_segmentation_04 = np.uint16(img_segmentation_04)
            
            img_segmentation_04 = correct_upper_level_segmentation(img_segmentation_01, img_segmentation_04, min_pixels = min_pixels_matching)
            
            model_trained_nuclei = models.CellposeModel(pretrained_model=path_model_nuclei_trained, gpu=flag_gpu)
            img_segmentation_nuclei, _, _ = model_trained_nuclei.eval(img_original_nuclei, diameter=None, channels= channels)        
            
            img_segmentation_01 = expand_labels(img_segmentation_01, distance=distance_expand_labels)
            img_segmentation_04 = expand_labels(img_segmentation_04, distance=distance_expand_labels)
            
            # del img_segmentation_01, img_segmentation_04
            #Remove segmentations in the mask
            
            if flag_masked:
                img_segmentation_01 = remove_overlapping_segmentation(img_segmentation_01, mask, min_pixels=10, percentage = 0)
                img_segmentation_04 = remove_overlapping_segmentation(img_segmentation_04, mask, min_pixels=10, percentage = 0)
                img_segmentation_nuclei = remove_overlapping_segmentation(img_segmentation_nuclei, mask, min_pixels=10, percentage = 0)        
                
            if flag_remove_objects_in_edge:
                #Remove nuclei that was matching cells in the edge
                _,_,nuclei_level_01_before_removing_edge_no_zeros = matching_label_pairs_perc(img_segmentation_nuclei, img_segmentation_01)
                _,_,nuclei_level_04_before_removing_edge_no_zeros = matching_label_pairs_perc(img_segmentation_nuclei, img_segmentation_04)
                
                img_excluded_region = img_segmentation_01 +  img_segmentation_04 + img_segmentation_nuclei
                
                #Get cells in the edges of the image
                img_segmentation_01 = remove_objects_in_edge(img_segmentation_01, margin = margin)
                img_segmentation_04 = remove_objects_in_edge(img_segmentation_04, margin = margin)
                img_segmentation_nuclei = remove_objects_in_edge(img_segmentation_nuclei, margin = margin)
                
                # If the cell is gone, the associated nuclei will be gone too
                for pair_nuclei_cell in nuclei_level_01_before_removing_edge_no_zeros:
                    label_nuclei = pair_nuclei_cell[0]
                    label_cell = pair_nuclei_cell[1]
                    if not (np.any(img_segmentation_01 == label_cell)): #If the cell is not there any more
                        img_segmentation_nuclei[img_segmentation_nuclei==label_nuclei] = 0
                
                for pair_nuclei_cell in nuclei_level_04_before_removing_edge_no_zeros:
                    label_nuclei = pair_nuclei_cell[0]
                    label_cell = pair_nuclei_cell[1]
                    if not (np.any(img_segmentation_04 == label_cell)): #If the cell is not there any more
                        img_segmentation_nuclei[img_segmentation_nuclei==label_nuclei] = 0
                
                img_excluded_new = img_segmentation_01 +  img_segmentation_04 + img_segmentation_nuclei
                
                img_excluded_region = np.logical_xor(img_excluded_region>0,img_excluded_new>0)
                # filename_excluded_region = os.path.join(folder_output_intermediate_output, sample_name + '_removed_in_edge.png')
                # cv2.imwrite(filename_excluded_region, img_excluded_region)
                
                del img_excluded_new, nuclei_level_01_before_removing_edge_no_zeros, nuclei_level_04_before_removing_edge_no_zeros
                
    
            # Histograms of cell sizes
            cell_props_level_01 = get_props_per_cell(img_segmentation_01)
            cell_props_level_04 = get_props_per_cell(img_segmentation_04)
            count_C1, edges_C1 = np.histogram(get_areas(cell_props_level_01), bins=n_bins)
            count_C4, edges_C4 = np.histogram(get_areas(cell_props_level_04), bins=n_bins)
            
            #Plot with the max value of any bin of the nuclei segmentation
            max_value_bin = np.max([np.max(count_C1),np.max(count_C4)])
    
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            
            plt.subplot(2, 1, 1)
            labels_x = [str(x) for x in np.uint32(np.array(edges_C1[:-1]) / 10.)]
            plt.bar(labels_x,count_C1)
            plt.xlabel('Pxs x 10e1', fontsize=7)
            plt.ylim(0, max_value_bin)
            plt.gca().set_title('Area level 1')
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
    
    
            plt.subplot(2, 1, 2)
            labels_x = [str(x) for x in np.uint32(np.array(edges_C4[:-1]) / 10.)]
            plt.bar(labels_x,count_C4)
            plt.ylim(0, max_value_bin)
            plt.gca().set_title('Area level 4') 
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
            plt.xlabel('Pxs x 10e1', fontsize=7)
            
            plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_area_cells_1_pre_cleaning.png'), dpi=400)
            
            plt.close('all')
            
            #Clean small cells
            _, good_cells_level_01_idx, noisy_cells_level_01_idx = detect_big_cells(cell_props_level_01, th_size=th_size_noisy_cells)
            _, good_cells_level_04_idx, noisy_cells_level_04_idx = detect_big_cells(cell_props_level_04, th_size=th_size_noisy_cells)
                    
            # Clean segmentation
            for idx_01 in noisy_cells_level_01_idx:
                img_segmentation_01[img_segmentation_01==idx_01] = 0
            for idx_04 in noisy_cells_level_04_idx:
                img_segmentation_04[img_segmentation_04==idx_04] = 0
            
            del cell_props_level_01, cell_props_level_04, count_C1, edges_C1, count_C4, edges_C4
                    
            edge_01 = filters.sobel(img_segmentation_01)
            edge_04 = filters.sobel(img_segmentation_04)
            edge_01 = edge_01 > 1.e-10
            edge_04 = edge_04 > 1.e-10
            edge_01 = dilation(edge_01, footprint=square(10))
            edge_04 = dilation(edge_04, footprint=square(10))
            
            sum_edges = np.uint8(edge_01) + np.uint8(edge_04)
            
            # del img_segmentation_01, img_segmentation_04
            
            # Histograms of cell sizes
            cell_props_level_01 = get_props_per_cell(img_segmentation_01)
            cell_props_level_04 = get_props_per_cell(img_segmentation_04)
            count_C1, edges_C1 = np.histogram(get_areas(cell_props_level_01), bins=n_bins)
            count_C4, edges_C4 = np.histogram(get_areas(cell_props_level_04), bins=n_bins)
            
            #Plot with the max value of any bin of the nuclei segmentation
            max_value_bin = np.max([np.max(count_C1),np.max(count_C4)])
            
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            
            plt.subplot(2, 1, 1)
            labels_x = [str(x) for x in np.uint32(np.array(edges_C1[:-1]) / 10.)]
            plt.bar(labels_x,count_C1)
            plt.xlabel('Pxs x 10e1', fontsize=7)
            plt.ylim(0, max_value_bin)
            plt.gca().set_title('Area level 1')
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
    
            plt.subplot(2, 1, 2)
            labels_x = [str(x) for x in np.uint32(np.array(edges_C4[:-1]) / 10.)]
            plt.bar(labels_x,count_C4)
            plt.ylim(0, max_value_bin)
            plt.gca().set_title('Area level 4') 
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
            plt.xlabel('Pxs x 10e1', fontsize=7)
            
            plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_area_cells_2_clean.png'), dpi=400)
            
            # Level 1
            img_segmentation_membrane_multiple_nuclei_level_01, img_segmentation_nuclei_in_same_cells_level_01, _ =\
                get_correspondance_segmentations_perc(img_segmentation_01, img_segmentation_nuclei)
                      
            # Level 4
            img_segmentation_membrane_multiple_nuclei_level_04, img_segmentation_nuclei_in_same_cells_level_04, _ =\
                get_correspondance_segmentations_perc(img_segmentation_04, img_segmentation_nuclei)    
            
            #Image composition
            img_composition = (img_original_nuclei / 2) + (img_original_Zo1 / 2)
            
            filename_composition = os.path.join(folder_output, sample_name + '_composite_nuclei_Zo1.png')
            img_composition_bgr = cv2.cvtColor((img_composition * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_composition, img_composition_bgr)
            
            if flag_remove_objects_in_edge:
                img_excluded_region_bgr = overlap_mask_over_image_rgb(img_composition_bgr, img_excluded_region, color_add = color_excluded_region)
                filename_excluded_region = os.path.join(folder_output_intermediate_output, sample_name + '_removed_in_edge.png')
                #img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename_excluded_region, img_excluded_region_bgr)
            
            ########### Background detection #################
            
            cells_and_nuclei_mask = (img_segmentation_01 > 0) | (img_segmentation_nuclei > 0) | (img_segmentation_04> 0)
            if flag_has_empty_space:
                if flag_cyto:
                    img_original_cyto_mask = median(img_original_cyto)
                    img_original_cyto_mask = img_original_cyto_mask > 0.06
                    cells_and_nuclei_mask = img_original_cyto_mask | (img_segmentation_01 > 0) | (img_segmentation_nuclei > 0) | (img_segmentation_04 > 0)
                
                # Compute that background
                structure_closing_background= disk(disk_diameter_background)
                largest_component_background, _, mask_closed = get_large_empty_spaces(cells_and_nuclei_mask, structure_closing_background)
            
            else:
                largest_component_background = np.zeros_like(img_segmentation_01)
                mask_closed = np.zeros_like(img_segmentation_01)
                
            
            ########## no_cell area candidates ############
            structure_closing_free_cell = disk(disk_diameter_free_cell)
            #Here it is to include or not level 04
            cells_mask = (img_segmentation_01 > 0) | largest_component_background | (img_segmentation_04 > 0)
            
            cells_mask_closed = ndimage.binary_closing(cells_mask, structure=structure_closing_free_cell, border_value=1)
            
            if flag_cyto:
                largest_component_no_cell, labeled_array_no_cell, _ = \
                    get_large_empty_spaces(cells_mask_closed, structure_closing_free_cell, th_size_px=th_size_no_cell_space)
            else:
                largest_component_no_cell, labeled_array_no_cell, _ = \
                    get_large_empty_spaces(cells_mask_closed, structure_closing_free_cell)
            
            if flag_remove_objects_in_edge:
                largest_component_no_cell = np.logical_and(largest_component_no_cell>0, np.logical_not(img_excluded_region))
            
            ######### Nuclei in no_cell zone #####
            
            _, _, matching_pairs_a_to_b_non_zero_level_04 = matching_label_pairs_perc(img_segmentation_nuclei, img_segmentation_04)
            img_segmentation_nuclei_excluding_in_cells = np.copy(img_segmentation_nuclei)
            for pair_nuclei_cell in matching_pairs_a_to_b_non_zero_level_04:
                img_segmentation_nuclei_excluding_in_cells[img_segmentation_nuclei_excluding_in_cells == pair_nuclei_cell[0]] = 0
            
            _, _, matching_pairs_a_to_b_non_zero = matching_label_pairs_perc(largest_component_no_cell, img_segmentation_nuclei_excluding_in_cells)
            
            list_nuclei_in_syncitial = [pair[1] for pair in matching_pairs_a_to_b_non_zero]
            img_segmentation_nuclei_in_syncitial = get_img_from_idx_cells(img_segmentation_nuclei, list_nuclei_in_syncitial)
            
            # Detect edges of the positive areas
            # Using Sobel edge detection
            edges_no_cell = filters.sobel(largest_component_no_cell.astype(float))
            edges_no_cell = edges_no_cell > 0.00001
            edges_background = filters.sobel(largest_component_background.astype(float))
            edges_background = edges_background > 0.00001
            
            # Nuclei and cell numbers
            n_cells_level_01 = len(np.unique(img_segmentation_01)) - 1 # Background is not a cell
            n_cells_level_04 = len(np.unique(img_segmentation_04)) - 1 # Background is not a cell
            n_cells_level_04_added = n_cells_level_04 - n_cells_level_01
            total_nuclei = len(np.unique(img_segmentation_nuclei)) - 1 #Background is not an object
            
            
            # Count the number of positive (True) pixels
            px_syncitial = np.sum(largest_component_no_cell)
            print("Number of no_cell pixels:", px_syncitial)
            
            plate = np.logical_and(np.logical_not(largest_component_background), np.logical_not(img_excluded_region > 0)) # Do not count exlusion
            img_plate_bgr = overlap_mask_over_image_rgb(img_composition_bgr, plate, color_add = color_plate)
            filename_plate = os.path.join(folder_output_intermediate_output, sample_name + '_valid_plate.png')
            cv2.imwrite(filename_plate, img_plate_bgr)
            px_plate = np.sum(plate)
            print("Number of plate pixels:", px_plate)
            
            ratio_syncitial_plate = float(px_syncitial) / float(px_plate + 0.0000001)
            nuclei_in_syncitial = len(np.unique(list_nuclei_in_syncitial))
            ratio_nuclei_syncitial = float(nuclei_in_syncitial) / float(total_nuclei + 0.0000001)
            print("total_nuclei:", total_nuclei)
                    
            n_cells_multiple_nuclei_level_01 = len(np.unique(img_segmentation_membrane_multiple_nuclei_level_01))-1 #0 is background
            n_cells_multiple_nuclei_level_04 = len(np.unique(img_segmentation_membrane_multiple_nuclei_level_04))-1 #0 is background
            
            n_nuclei_sharing_membrane_level_01 = len(np.unique(img_segmentation_nuclei_in_same_cells_level_01))-1 #0 is background
            n_nuclei_sharing_membrane_level_01 = 0 if n_cells_multiple_nuclei_level_01 == 0 else n_nuclei_sharing_membrane_level_01 #if no multinuclei cells, no nuclei sharing cells
            n_nuclei_sharing_membrane_level_04 = len(np.unique(img_segmentation_nuclei_in_same_cells_level_04))-1 #0 is background
            n_nuclei_sharing_membrane_level_04 = 0 if n_cells_multiple_nuclei_level_04 == 0 else n_nuclei_sharing_membrane_level_04 #if no multinuclei cells, no nuclei sharing cells
            
            total_nuclei_in_multinuclei_cells_and_syncitial = n_nuclei_sharing_membrane_level_04 + nuclei_in_syncitial
            fusion_index = total_nuclei_in_multinuclei_cells_and_syncitial / (total_nuclei + 0.0000001) # ratio_nuclei_in_multinuclei_cells_and_syncitial
            ratio_nuclei_in_multinuclei_cells = n_nuclei_sharing_membrane_level_04 / (total_nuclei + 0.0000001)
            
            # Normalization to compute the slope
            n_cells_level_01_norm = n_cells_level_01 / n_cells_level_04
            n_cells_level_04_norm = n_cells_level_04 / n_cells_level_04
            
            ratio_cells_multiple_nuclei_level_04 = n_cells_multiple_nuclei_level_04 / (n_cells_level_04 + 0.0000001)
            
            n_nuclei_in_uninucleited_cells = total_nuclei - (nuclei_in_syncitial + n_nuclei_sharing_membrane_level_04)
            
            # Fit linear regression model
            vector_n_cells = np.array([n_cells_level_01_norm, n_cells_level_04_norm])
            levels_cells = np.array([1, 4])
            linear_results = stats.linregress(levels_cells, vector_n_cells)
            
            ratio_firm_over_uninucleate = n_cells_level_01 / (n_nuclei_in_uninucleited_cells + 0.0000001)
            ratio_porous_over_uninucleate = n_cells_level_04_added / (n_nuclei_in_uninucleited_cells + 0.0000001)
            
            #Lucas stats to compare with Myriam stats
            
            ratio_firm_over_total_cells = n_cells_level_01 / n_cells_level_04
            ratio_nuclei_in_uninucleited_cells_over_total_nuclei = n_nuclei_in_uninucleited_cells / total_nuclei
            ratio_nuclei_in_multinucleated_cells_over_total_nuclei = n_nuclei_sharing_membrane_level_04 / total_nuclei
            
            #####################################################################################################
            ##################################   GENERATE OUTPUTS    ############################################
            #####################################################################################################
            
            plt.figure()
            plt.plot(levels_cells, vector_n_cells, 'o', label='n cells')
            plt.plot(levels_cells, linear_results.intercept + linear_results.slope*levels_cells, 'r', label='fitted line')
            plt.xlabel('Zo1 strength')
            plt.ylabel('n cells norm')
            plt.ylim(0, 1.1)
            plt.xlim(0,max(levels_cells)+1)        
            plt.legend()
            plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_n_cells_fitting.png'), dpi=400)
            
            #                 "SampleName", "fusion_index", "RatioFirmCellsVsUninucleateCells(E/D)", "RatioPorousCellsVsUninucleateCells([F-E]/D)"
            df.loc[len(df)] = [sample_name, fusion_index,    ratio_firm_over_uninucleate,             ratio_porous_over_uninucleate, \
                               # "RatioNucleiInMultinucleateCellsVsTotalNuclei(C/A)"
                               ratio_nuclei_in_multinucleated_cells_over_total_nuclei, \
            #                 "empty col", "RatioFirmCellsVsTotalCells(E/F)", "RatioNucleiInUnicellsVsTotalNuclei(D/A)"
                               None,        ratio_firm_over_total_cells,      ratio_nuclei_in_uninucleited_cells_over_total_nuclei]
            txt_output = os.path.join(folder_output, sample_name + '_summary.txt')
            f = open(txt_output, "w")
            
            f.write(sample_name + '\n')
            
            f.write('------------------------------------------------------- \n')        
            f.write('NUCLEI STATS \n\n')
            f.write('Total nuclei (A): ' + str( total_nuclei) + '\n')
            f.write('Syncytial nuclei (B): ' + str( nuclei_in_syncitial) + '\n')
            f.write('Nuclei in multinucleate cells (C): ' + str( n_nuclei_sharing_membrane_level_04) + '\n')
            f.write('Nuclei in uninucleate cells (D = A-(B+C)): ' + str( n_nuclei_in_uninucleited_cells ) + '\n')
            f.write('Total nuclei in multinucleate cells (at level 4) and Syncytial nuclei (B+C): ' + str( total_nuclei_in_multinuclei_cells_and_syncitial) + '\n')
            f.write('Fusion index ([B+C]/A): ' + str( fusion_index) + '\n')
            
            f.write('\n------------------------------------------------------- \n')
            f.write('CELL STATS \n\n')
            f.write('All cells (F) (Level 04): ' + str( n_cells_level_04) + '\n')        
            f.write('Firm membrane (E) (Level_01): ' + str( n_cells_level_01) + '\n')
            f.write('Porous membrane (F-E) (Level_04 - Level_01): ' + str( n_cells_level_04_added) + '\n')
    
            f.write('Ratio of firm cells over total uninucleate cells (E/D): ' + str(ratio_firm_over_uninucleate ) + '\n')
            f.write('Ratio of porous cells over total uninucleate cells ([F-E]/D): ' + str(ratio_porous_over_uninucleate ) + '\n')
            
            f.write('\n------------------------------------------------------- \n')
            f.write('Total cells per level:\n')
            f.write('All cells (F) (Level 04): ' + str( n_cells_level_04) + '\n')
            f.write('Firm membrane (E) (Level 01): ' + str( n_cells_level_01) + '\n')
            f.write('Ratio of firm cells over total cells (E/F): ' + str(ratio_firm_over_total_cells ) + '\n')
            f.write('------------------------------------------------------- \n')
            f.write('Normalization:\n')
            f.write('Level_04: ' + str( n_cells_level_04_norm) + '\n')
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
            f.write('Level 04: ' + str( n_cells_multiple_nuclei_level_04) + '\n')
            f.write('Nuclei sharing cells level 04 (C): ' + str( n_nuclei_sharing_membrane_level_04) + '\n')
            f.write('------------------------------------------------------- \n')
            f.write('Total nuclei (A): ' + str( total_nuclei) + '\n')
            f.write('Ratio Syncytial nuclei: ' + str( ratio_nuclei_syncitial) + '\n')
            f.write('Ratio nuclei in multinuclei cells (B/A): ' + str( ratio_nuclei_in_multinuclei_cells) + '\n')
            f.write('Ratio multinuclei cells vs total cells: ' + str( ratio_cells_multiple_nuclei_level_04) + '\n')
            f.write('Ratio nuclei in uni-cells vs total nuclei (D/A): ' + str( ratio_nuclei_in_uninucleited_cells_over_total_nuclei) + '\n')
            f.write('\n')
            f.write('Number of pixels in syncitial area: ' + str(px_syncitial) + '\n')
            f.write('Number of pixels in plate (after removing background): ' + str( px_plate) + '\n')
            f.write('Ratio syncitial area - pxs in plate: ' + str( ratio_syncitial_plate) + '\n')
            f.write('------------------------------------------------------- \n')
            f.close()
            
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
            plt.subplot(2, 2, 1)
            plt.imshow(img_original_Zo1,cmap='gray')
            plt.gca().set_title('Zo1', fontsize=fontsize)
            plt.subplot(2, 2, 2)
            plt.imshow(sum_edges)
            plt.gca().set_title('Edges', fontsize=fontsize)
            plt.subplot(2, 2, 3)
            plt.imshow(img_original_nuclei,cmap='gray')
            plt.gca().set_title('Nuclei', fontsize=fontsize)
            plt.subplot(2, 2, 4)
            plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
            plt.gca().set_title('Nuclei seg', fontsize=fontsize)
            
            
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots(3, 3, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
            plt.subplot(3, 3, 1)
            plt.imshow(img_composition,cmap='gray')
            plt.gca().set_title('Composition', fontsize=fontsize)
            
            plt.subplot(3, 3, 2)
            plt.imshow(sum_edges)
            plt.gca().set_title('Edges', fontsize=fontsize)
            
            plt.subplot(3, 3, 3)
            plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
            plt.gca().set_title('Nuclei: ' + str(total_nuclei), fontsize=fontsize)
            
            plt.subplot(3, 3, 4)
            plt.imshow(img_segmentation_01)
            plt.gca().set_title('Cell well def: ' + str(n_cells_level_01), fontsize=fontsize)
            
            plt.subplot(3, 3, 5)
            plt.imshow(img_segmentation_membrane_multiple_nuclei_level_01,cmap='gist_ncar')
            plt.gca().set_title('Multi-nuclei cells well def: ' + str(n_cells_multiple_nuclei_level_01), fontsize=fontsize)
            
            plt.subplot(3, 3, 6)
            plt.imshow(img_segmentation_nuclei_in_same_cells_level_01,cmap='gist_ncar')
            plt.gca().set_title('Nuclei in same cells well def: '+ str( n_nuclei_sharing_membrane_level_01), fontsize=fontsize)
            
            
            plt.subplot(3, 3, 7)
            plt.imshow(img_segmentation_04)
            plt.gca().set_title('All cells: ' + str(n_cells_level_04), fontsize=fontsize)
            
            plt.subplot(3, 3, 8)
            plt.imshow(img_segmentation_membrane_multiple_nuclei_level_04,cmap='gist_ncar')
            plt.gca().set_title('Multi-nuclei cells: ' + str(n_cells_multiple_nuclei_level_04), fontsize=fontsize)
            
            plt.subplot(3, 3, 9)
            plt.imshow(img_segmentation_nuclei_in_same_cells_level_04,cmap='gist_ncar')
            plt.gca().set_title('Nuclei in same cells: ' + str( n_nuclei_sharing_membrane_level_04), fontsize=fontsize)
            
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            # plt.show()
            
            plt.savefig(os.path.join(folder_output_intermediate_output, sample_name + '_sharpness_cells.png'), dpi=400)
            
            #Output
            img_segmentation_nuclei_in_syncitial = draw_roi_over_image(img_original_nuclei, img_segmentation_nuclei_in_syncitial)
            img_composition_syncitial_edge = draw_mask_over_image(img_composition, edges_no_cell, color_mask = [255, 0, 0])
            img_composition_syncitial_edge = draw_mask_over_image_rgb(img_composition_syncitial_edge, edges_background, color_mask = [0, 40, 40])
            
            filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_syncitialR_backC.png')
            img_composition_syncitial_edge_bgr = cv2.cvtColor(img_composition_syncitial_edge, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_syncitial_back, img_composition_syncitial_edge_bgr)
            
            #largest_component_no_cell
            img_composition_syncitial = overlap_mask_over_image_rgb(img_composition_syncitial_edge, largest_component_no_cell, color_add = color_add_free_nuclei) # [50, 0, 0]
            filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_syncitialR_backC_fill.png')
            img_composition_syncitial_edge_bgr = cv2.cvtColor(img_composition_syncitial, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_syncitial_back, img_composition_syncitial_edge_bgr)
            
            #largest_component_no_cell
            img_composition_syncitial = overlap_mask_over_image_rgb(img_composition_syncitial, largest_component_background, color_add = color_add_back) # [0, 40, 40]
            filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_syncitialR_backC_fill_2.png')
            img_composition_syncitial_edge_bgr = cv2.cvtColor(img_composition_syncitial, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_syncitial_back, img_composition_syncitial_edge_bgr)
            
            del img_composition_syncitial, filename_syncitial_back, img_composition_syncitial_edge_bgr
            
            # For debug
            if flag_cyto:
                img_composition_cyto = draw_mask_over_image(img_composition, img_original_cyto_mask, color_mask = [255, 0, 255])
                filename_syncitial_back = os.path.join(folder_output_intermediate_output, sample_name + '_cyto.png')
                img_composition_cyto_bgr = cv2.cvtColor(img_composition_cyto, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename_syncitial_back, img_composition_cyto_bgr)
            
            # Draw edges cells
            # For a correct overlapping, from the weakest edge to the stronger ones
            
            img_composition_edges = draw_mask_over_image(img_original_Zo1, edge_01, color_mask = color_edges_01) # [255, 17, 0]
            filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_level01.png')
            img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
            
            # Level 04 in the composition to see multi-nuclei cells
            img_composition_level04 = draw_mask_over_image(img_composition, edge_04, color_mask = color_edges_04) # [0, 40, 255]
            filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_level04_in_composition.png')
            img_composition_edges_bgr = cv2.cvtColor(img_composition_level04, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
            
            if flag_masked:
                # Level 04 in the composition to see multi-nuclei cells
                img_composition_mask = draw_mask_over_image(img_composition, mask, color_mask = color_add_multiple_nuclei) # [0, 40, 255]
                filename_composition_mask_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_maskCyan.png')
                img_composition_mask_bgr = cv2.cvtColor(img_composition_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename_composition_mask_bgr, img_composition_mask_bgr)
                
            
            # Composition of images
            
            img_composition_edges = draw_mask_over_image(img_original_Zo1, edge_04, color_mask = color_rainbow_sum_1) # [0, 40, 255]
            filename_edges_bgr = os.path.join(folder_output, sample_name + '_edges_level04.png')
            img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
            
            img_composition_edges = draw_mask_over_image_rgb(img_composition_edges, edge_01, color_mask = color_rainbow_sum_4) # [255, 17, 0]
            
            filename_edges_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_edges_firmR_porousG.png')
            img_composition_edges_bgr = cv2.cvtColor(img_composition_edges, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_edges_bgr, img_composition_edges_bgr)
            
            sum_cell_seg = np.uint8(img_segmentation_01 > 0) + np.uint8(img_segmentation_04 > 0)
            img_composition_sum_cell = draw_mask_over_image(img_composition,                sum_cell_seg == 1, color_mask = color_rainbow_sum_1) # [0, 40, 255]
            img_composition_sum_cell = draw_mask_over_image_rgb(img_composition_sum_cell,   sum_cell_seg == 2, color_mask = color_rainbow_sum_4) # [255, 17, 0]
            
            filename_sum_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_rainbow_firmR_porousG.png')
            img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
            
            #img_composition_sum_cell = draw_mask_over_image(img_composition, edge_04, color_mask = [0, 0, 100])
            img_composition_sum_cell = (img_composition * 255).astype(np.uint8)
            img_composition_sum_cell = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_GRAY2BGR)
            img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, sum_cell_seg == 1, color_add = color_add_rainbow_1) # [0, 0, 80]
            img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, sum_cell_seg == 2, color_add = color_add_rainbow_4) # [80, 0, 0]
            filename_sum_bgr = os.path.join(folder_output, sample_name + '_firmB_porousG.png')
            img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
            
            img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, largest_component_background>0, color_add = color_add_back) # [0, 40, 40]
            filename_sum_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_rainbow_backCyan_firmB_porousG.png')
            img_composition_sum_bgr = cv2.cvtColor(img_composition_sum_cell, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_sum_bgr, img_composition_sum_bgr)
            
            
            # img_composition_sum_cell = overlap_mask_over_image_rgb(img_composition_sum_cell, largest_component_no_cell>0, color_add = [80, 0, 80])
            img_composition_sum_cell = \
                overlap_mask_over_image_rgb(img_composition_sum_cell, largest_component_no_cell, color_add = [80, 0, 80])
            filename_sum_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_rainbow_backCyan_syncitialPink_firmB_porousG.png')
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
            filename_nuclei_bgr = os.path.join(folder_output, sample_name + '_nucleiC_segLevel04B_syncitialR.png')
            img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_nuclei_bgr, img_composition_nuclei_bgr)
            
            img_composition_nuclei = overlap_mask_over_image_rgb(img_composition_nuclei, img_segmentation_membrane_multiple_nuclei_level_04>0, color_add = color_add_multiple_nuclei) # [80, 80, 10]
            filename_nuclei_bgr = os.path.join(folder_output, sample_name + '_nucleiC_segLevel04B_syncitialR_multinucleiY.png')
            img_composition_nuclei_bgr = cv2.cvtColor(img_composition_nuclei, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_nuclei_bgr, img_composition_nuclei_bgr)
            
            # Multinuclei in other levels
            #Draw nuclei in composition
            img_composition_nuclei = draw_mask_over_image(img_composition, img_segmentation_nuclei > 0, color_mask = color_nuclei) # [0, 150, 150]
            
            img_composition_nuclei = draw_mask_over_image(img_composition, img_segmentation_nuclei > 0, color_mask = color_nuclei) # [0, 150, 150]
            
            img_composition_nuclei = draw_mask_over_image(img_composition, img_segmentation_nuclei > 0, color_mask = color_nuclei) # [0, 150, 150]
            img_composition_nuclei = draw_mask_over_image_rgb(img_composition_nuclei, edge_01, color_mask = color_edges_01)
            img_composition_nuclei = overlap_mask_over_image_rgb(img_composition_nuclei, img_segmentation_membrane_multiple_nuclei_level_01>0, color_add = color_add_multiple_nuclei) # [80, 80, 10]
            filename_nuclei_bgr = os.path.join(folder_output_intermediate_output, sample_name + '_nucleiC_segLevel01R.png')
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
            
            # Build vectors for boxplots
            
            if flag_wt:
                vector_FusionIndex_wt.append(fusion_index)
                vector_RatioFirmCellsVsUniCells_wt.append(ratio_firm_over_uninucleate)
                vector_RatioPorousCellsVsUniCells_wt.append(ratio_porous_over_uninucleate)
                vector_RatioFirmCellsVsTotalCells_wt.append(ratio_firm_over_total_cells)
                vector_RatioNucleiInUnicellsVsTotalNuclei_wt.append(ratio_nuclei_in_uninucleited_cells_over_total_nuclei)
                vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_wt.append(ratio_nuclei_in_multinucleated_cells_over_total_nuclei)
            elif flag_mut:
                vector_FusionIndex_mut.append(fusion_index)
                vector_RatioFirmCellsVsUniCells_mut.append(ratio_firm_over_uninucleate)
                vector_RatioPorousCellsVsUniCells_mut.append(ratio_porous_over_uninucleate)
                vector_RatioFirmCellsVsTotalCells_mut.append(ratio_firm_over_total_cells)
                vector_RatioNucleiInUnicellsVsTotalNuclei_mut.append(ratio_nuclei_in_uninucleited_cells_over_total_nuclei)
                vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_mut.append(ratio_nuclei_in_multinucleated_cells_over_total_nuclei)
            elif flag_het:
                vector_FusionIndex_het.append(fusion_index)
                vector_RatioFirmCellsVsUniCells_het.append(ratio_firm_over_uninucleate)
                vector_RatioPorousCellsVsUniCells_het.append(ratio_porous_over_uninucleate)
                vector_RatioFirmCellsVsTotalCells_het.append(ratio_firm_over_total_cells)
                vector_RatioNucleiInUnicellsVsTotalNuclei_het.append(ratio_nuclei_in_uninucleited_cells_over_total_nuclei)
                vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_het.append(ratio_nuclei_in_multinucleated_cells_over_total_nuclei)
                
            del cells_and_nuclei_mask, largest_component_background, mask_closed, image_input_nuclei, image_input_Zo1
            del image_input_cyto, img_composition_sum_cell, img_composition_nuclei_bgr
     
    # Save table
    path_csv = os.path.join(folder_images, 'Metrics_dataset.csv')
    df.to_csv(path_csv, index=False)
    
    # Plot box plots
    
    def plot_and_save_boxplot(vector_wt,vector_mut,vector_het,variable_name, first_part = 'Boxplot_'):
    
        # Create boxplot
        plt.figure(figsize=(6, 4))
        plt.boxplot([vector_wt,vector_mut,vector_het], patch_artist=True, 
                    boxprops=dict(facecolor="lightcoral"),  # KO in red
                    medianprops=dict(color="black"))
        
        # Customization
        plt.xticks([1, 2, 3], ["WT", "MUT", "HET"])  # X-axis labels
        plt.ylabel(variable_name)
        plt.title(variable_name)
        plt.savefig(os.path.join(folder_images, first_part + variable_name + '.png'), dpi=400)
        plt.close('all')
    
    plot_and_save_boxplot(vector_FusionIndex_wt,vector_FusionIndex_mut,vector_FusionIndex_het,'Fusion_index')
    plot_and_save_boxplot(vector_RatioFirmCellsVsUniCells_wt,vector_RatioFirmCellsVsUniCells_mut,vector_RatioFirmCellsVsUniCells_het,'RatioFirmCellsVsUniCells')
    plot_and_save_boxplot(vector_RatioPorousCellsVsUniCells_wt,vector_RatioPorousCellsVsUniCells_mut,vector_RatioPorousCellsVsUniCells_het,'RatioPorousCellsVsUniCells')
    plot_and_save_boxplot(vector_RatioFirmCellsVsTotalCells_wt,vector_RatioFirmCellsVsTotalCells_mut,vector_RatioFirmCellsVsTotalCells_het,'RatioFirmCellsVsTotalCells')
    plot_and_save_boxplot(vector_RatioNucleiInUnicellsVsTotalNuclei_wt,vector_RatioNucleiInUnicellsVsTotalNuclei_mut,vector_RatioNucleiInUnicellsVsTotalNuclei_het,'RatioNucleiInUnicellsVsTotalNuclei')
    plot_and_save_boxplot(vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_wt,vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_mut,vector_RatioNucleiInMultinucleateCellsVsTotalNuclei_het,'RatioNucleiInMultinucleateCellsVsTotalNuclei')
    
    plt.close('all')

if __name__ == "__main__":
    main()
