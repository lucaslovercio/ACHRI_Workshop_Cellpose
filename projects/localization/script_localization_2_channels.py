#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:37:27 2023

@author: lucas
"""

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

###################################   PARAMETERS   #########################

folder_images = '' #In Windows, place an r before the ''
folder_models = '' #In Windows, place an r before the ''
model_nuclear       = 'HEK_nuclei_60x_model_cyto2_diam_50_ji_0.869.794803' #In Windows, place an r before the ''
model_channel_2 = 'HEK_second_channel_60x_model_cyto2_diam100_ji_0.862.339044' #In Windows, place an r before the ''
list_classes = ['ctrl', 'HBSS'] # These substrings needs to be present in the filenames of the tiff files

min_pixels_matching = 500 # If cells are small, reduce this number

#Can be changed
flag_normalize = True # Recommended, and it has to be same used for Cellpose models
flag_gpu = False # If not sure, use False
flag_use_median = False # Median is recommended due to the (speckle) noise of expression signals
channels = [[0,0]] # Same channels as training for the Cellpose models

##############################################################################
##############################################################################
##############################################################################
##############################################################################


import cv2
import numpy as np
from cellpose import models
import matplotlib.pyplot as plt
from quantify_segmentation import get_props_per_cell, \
        matching_label_pairs, get_expr_from_labels, get_labels, get_centroids,\
            plot_expressions, plot_expressions_labelled, get_img_with_ids
from aux_functions.functionPercNorm import functionPercNorm
import os
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list
import pandas as pd

def main():
    
    path_model_nuclei_trained  = os.path.join(folder_models,model_nuclear)
    path_model_second_trained  = os.path.join(folder_models,model_channel_2)
    
    files_tiff = []
    
    for file in os.listdir(folder_images):
        if file.endswith(".tif") or file.endswith(".tiff"):
            files_tiff.append(file)
    
    list_all_nuclei_expression = []
    list_all_cito_expression = []
    list_all_label_image = []
    list_all_filename_image = []
    list_all_nuclei_id = []
    list_all_cito_id = []
    list_all_nuclei_area = []
    list_all_cito_area = []
    
    for filename_tiff in files_tiff:
        path_tiff = os.path.join(folder_images, filename_tiff)
        
        label_image = next((x for x in list_classes if x in filename_tiff), None)
        
        print('------------------------------------')
        print('Analyzing: ' + filename_tiff)    
        
        list_channels = read_multipage_tiff_as_list(path_tiff)
        img_nuclei_original = list_channels[0]
        img_expression_original = list_channels[1]
        
        if flag_normalize:
            img_expression_original = functionPercNorm( np.single(img_expression_original))
        
        #Segment nuclei
        model_trained_nuclei = models.CellposeModel(pretrained_model=path_model_nuclei_trained, gpu=flag_gpu)
        
        if flag_normalize:
            img_nuclei_original = functionPercNorm( np.single(img_nuclei_original))
        
        #Segment nuclei in image
        img_segmentation_nuclei, _, _ = model_trained_nuclei.eval(img_nuclei_original, diameter=None, channels= channels)
        del model_trained_nuclei
        
        model_trained_second = models.CellposeModel(pretrained_model=path_model_second_trained, gpu=flag_gpu)
        
        img_segmentation_second_channel, _, _ = model_trained_second.eval(img_expression_original, diameter=None, channels= channels)
        
        #Get centroids and shape descriptors
        cell_props_nuclei = get_props_per_cell(img_segmentation_nuclei)
        labels_nuclei = get_labels(cell_props_nuclei)
        centroids_nuclei = get_centroids(cell_props_nuclei)
        
        img_with_nuclei_ids = get_img_with_ids(np.uint8(img_nuclei_original * 255), centroids_nuclei, labels_nuclei)
        img_nuclei_ids_output        = path_tiff + '_nuclei_id.png'
        cv2.imwrite(img_nuclei_ids_output, img_with_nuclei_ids)
        #Save nuclei segmentation
        segmentation_nuclei_output        = path_tiff + '_seg.png'
        cv2.imwrite(segmentation_nuclei_output, img_segmentation_nuclei)
        
        # Second channel
        cell_props_second_channel = get_props_per_cell(img_segmentation_second_channel)
        labels_cito = get_labels(cell_props_second_channel)
        centroids_cito = get_centroids(cell_props_second_channel)
        img_with_cito_ids = get_img_with_ids(np.uint8(img_expression_original * 255), centroids_cito, labels_cito)
        img_cito_ids_output        = path_tiff + '_cito_id.png'
        cv2.imwrite(img_cito_ids_output, img_with_cito_ids)
        
        #Matching nuclei and cells
        
        _, _, matching_pairs_a_to_b_non_zero = matching_label_pairs(img_segmentation_nuclei, img_segmentation_second_channel, min_pixels=min_pixels_matching)
        
        if (matching_pairs_a_to_b_non_zero is not None) and len(matching_pairs_a_to_b_non_zero) > 0:
            
            # Get expression
            
            img_segmentation_citoplasm_minus_nucleus = np.copy(img_segmentation_second_channel)
            img_segmentation_citoplasm_minus_nucleus[img_segmentation_nuclei>0] = 0
            
            left, right = zip(*matching_pairs_a_to_b_non_zero)
            labels_nuclei_matched = list(left)
            labels_cells_matched = list(right)
            del labels_nuclei
            
            list_nuclei_expression = get_expr_from_labels(labels_nuclei_matched, img_segmentation_nuclei, img_expression_original, flag_use_median = flag_use_median)
            list_cito_expression = get_expr_from_labels(labels_cells_matched, img_segmentation_citoplasm_minus_nucleus, img_expression_original, flag_use_median = flag_use_median)
            
            # Extend the list of all the folders
            list_all_nuclei_expression.extend(list_nuclei_expression)
            list_all_cito_expression.extend(list_cito_expression)
            list_all_label_image.extend([label_image] * len(list_nuclei_expression))
            list_all_filename_image.extend([filename_tiff] * len(list_nuclei_expression))
            list_all_nuclei_id.extend(labels_nuclei_matched)
            list_all_cito_id.extend(labels_cells_matched)
            
            areas_nuclei_matched = []
            for id_nuclei in labels_nuclei_matched:
                for cell in cell_props_nuclei:
                    if cell.label == id_nuclei:
                        areas_nuclei_matched.append(cell.area)
                        break
            
            list_all_nuclei_area.extend(areas_nuclei_matched)
            
            areas_cito_matched = []
            for id_cito in labels_cells_matched:
                for cell in cell_props_second_channel:
                    if cell.label == id_cito:
                        areas_cito_matched.append(cell.area)
                        break
            
            # Removing nuclei area
            areas_cito_matched = list(np.array(areas_cito_matched) - np.array(areas_nuclei_matched))
            
            list_all_cito_area.extend(areas_cito_matched)
            
            fontsize = 8
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            #fig = plt.figure()
            fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
            
            plt.subplot(2, 3, 1)
            plt.imshow(img_expression_original,cmap='gray')
            plt.gca().set_title('Expression', fontsize=fontsize)
            plt.subplot(2, 3, 2)
            plt.imshow(img_segmentation_second_channel,cmap='gist_ncar')
            plt.gca().set_title('Second channel', fontsize=fontsize)
            
            plt.subplot(2, 3, 4)
            plt.imshow(img_nuclei_original,cmap='gray')
            plt.gca().set_title('Nuclei original', fontsize=fontsize)
            plt.subplot(2, 3, 5)
            plt.imshow(img_segmentation_nuclei,cmap='gist_ncar')
            plt.gca().set_title('Segmentation - Total nuclei:' + str(len(cell_props_nuclei)), fontsize=fontsize)
            
            plt.subplot(2, 3, 6)
            plt.imshow(img_segmentation_citoplasm_minus_nucleus,cmap='gist_ncar')
            plt.gca().set_title('cito minus nuclei', fontsize=fontsize)
            
            png_output_segmentations = path_tiff + '_segmentations.png'
            plt.savefig(png_output_segmentations, dpi=800)
            
            plot_expressions(list_nuclei_expression, list_cito_expression, 'expression', \
                             label_x = 'expr in nuclei', label_y = 'expr in cito', figsize = 4, flag_show = False)
            png_output_expressions = path_tiff + '_expressions.png'
            plt.savefig(png_output_expressions, dpi=800)
            
            plt.close('all')
    
    
    plot_expressions_labelled(list_all_nuclei_expression, list_all_cito_expression, list_all_label_image, 'expression', \
                              label_x = 'expr in nuclei', label_y = 'expr in cito', figsize = 8, flag_show = False)
    png_output_expressions = os.path.join(folder_images, 'expressions.png')
    plt.savefig(png_output_expressions, dpi=800)
    
    str_mean_median = 'median' if flag_use_median else 'mean'
    df = pd.DataFrame({
        'Image': list_all_filename_image,
        'Treatment': list_all_label_image,
        'Nucleus id': list_all_nuclei_id,
        'Nucleus area': list_all_nuclei_area,
        'Nucleus '+str_mean_median+' exp': list_all_nuclei_expression,
        'Nucleus '+str_mean_median+' x area' : np.array(list_all_nuclei_area) * np.array(list_all_nuclei_expression),
        'Cito id': list_all_cito_id,
        'Cito area': list_all_cito_area,
        'Cito '+str_mean_median+' exp': list_all_cito_expression,
        'Cito '+str_mean_median+' x area' : np.array(list_all_cito_area) * np.array(list_all_cito_expression),
    })
    
    # save to csv
    csv_output_expressions = os.path.join(folder_images, 'expressions.csv')
    df.to_csv(csv_output_expressions, index=False)

    
    
if __name__ == "__main__":
    main()
