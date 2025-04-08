#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:32:13 2024

@author: lucas
"""

###################################   PARAMETERS   #########################


folder_images = '/mnt/DATA/ACHRI/2024-03 Jack lab/Nuclei_mitosis/'  #In Windows, place an r before the ''


path_model_all_nuclei       = ''   #In Windows, place an r before the ''
path_model_regular_nuclei   = ''   #In Windows, place an r before the ''
path_model_mitotic_nuclei   = ''   #In Windows, place an r before the ''
path_model_xy_mitotic       = ''   #In Windows, place an r before the ''
path_model_tubulin_mitosis  = ''   #In Windows, place an r before the ''


pattern_channel_dapi = '_c0'
pattern_channel_tubulin = '_c1'


#Can be changed
flag_normalize = True
flag_gpu = False
channels = [[0,0]] #Same channels as training

# To remove small objects (such as chromocenters)
th_obj = 350

# For matching between cells and with tubulin segmentation
fraction_cell = 1/2
min_area_overlap = th_obj * fraction_cell
percentage_overlap_nuclei = 0.4

###################################   PARAMETERS   #########################

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from cellpose import models
from cellpose.io import imread
from aux_functions.functionPercNorm import functionPercNorm, get_one_channel
from aux_functions.filepath_management import get_sample_name, get_image_filenames
import numpy as np
import matplotlib.pyplot as plt
from quantify_segmentation import get_props_per_cell, matching_label_pairs,\
    remove_overlapping_segmentation, get_overlapping_segmentation
from skimage import measure
import cv2
from scipy import ndimage
from aux_functions.draw_roi_over_image import draw_roi_over_image

if os.path.exists(path_model_regular_nuclei) and os.path.exists(path_model_mitotic_nuclei) and\
    os.path.exists(path_model_xy_mitotic) and os.path.exists(path_model_tubulin_mitosis) and\
        os.path.exists(path_model_all_nuclei):
    print('All models found')
else:
    print('A model was not found. Stopping execution.')
    sys.exit(1)
            
if os.path.exists(folder_images) and os.path.isdir(folder_images):
    print(f"The folder '{folder_images}' exists.")
else:
    print('Folder not found. Stopping execution.')
    sys.exit(1)
    
image_filenames = get_image_filenames(folder_images, substring_pattern = pattern_channel_dapi)    

def get_tubulin_vertices(greyscale_image, img_segmentation):
    # List to store sub-images
    sub_images = []
    labels = []
    number_vertices = []
    object_labels = np.unique(img_segmentation)
    object_labels = object_labels[object_labels>0]
    img_vertices = np.zeros(img_segmentation.shape, dtype=np.uint16)

    for label in object_labels:
        # Create a mask for the current object
        mask = img_segmentation == label
        
        # Find the bounding box coordinates for the current object
        y_indices, x_indices = np.where(mask)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        # Extract the sub-image using the bounding box coordinates
        sub_image = np.copy(greyscale_image[y_min:y_max, x_min:x_max])
        sub_image_mask = mask[y_min:y_max, x_min:x_max]
        
        #To not catch parts of other cells/tubulin
        sub_image[sub_image_mask==0] = 0
        
        # Compute vertices
        h, w = sub_image.shape
        center_x, center_y = w // 2, h // 2
        
        radius = min(h, w) // 4
        # Create a circular mask
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        circular_mask = dist_from_center <= radius
        
        # Apply the mask to set the central circular region to black
        sub_image[circular_mask] = 0
        
        
        sub_image = ndimage.median_filter(sub_image, size=3)
        sub_image = ndimage.median_filter(sub_image, size=3)
        sub_image = ndimage.median_filter(sub_image, size=3)
        sub_image = functionPercNorm( sub_image, [0.01, 0.99])
        
        binary_image = sub_image>0.6
        labelled, n_objects = ndimage.label(binary_image)
        
        number_vertices.append(n_objects)
        img_vertices[mask] = n_objects
        # Add the sub-image to the list
        sub_images.append(sub_image)
        labels.append(label)
        
    return img_vertices, labels, number_vertices, sub_images
    

# Print the list of image filenames
for filename_nuclei in image_filenames:
    print('Analyzing: ' + filename_nuclei)
    
    sample_name = get_sample_name(filename_nuclei)
    folder_output = os.path.join(folder_images, sample_name)
    os.makedirs(folder_output, exist_ok=True)
    
    
    input_nuclei    = os.path.join(folder_images,filename_nuclei)
    input_tubulin  = os.path.join(folder_images,filename_nuclei.replace(pattern_channel_dapi, pattern_channel_tubulin))

    plt.close('all')
    
    img_nuclei_original = imread(input_nuclei)
    img_nuclei_original = get_one_channel(img_nuclei_original)
    
    img_tubulin_original = imread(input_tubulin)
    img_tubulin_original = get_one_channel(img_tubulin_original)
    
    del input_nuclei, input_tubulin
    
    if flag_normalize:
        img_nuclei_original = functionPercNorm( np.single(img_nuclei_original))
        img_tubulin_original = functionPercNorm( np.single(img_tubulin_original))
    
    model_trained_all_nuclei = models.CellposeModel(pretrained_model=path_model_all_nuclei, gpu=flag_gpu)
    img_segmentation_all_nuclei, flows, styles = model_trained_all_nuclei.eval(img_nuclei_original, diameter=None, channels= channels)
    
    #Remove small objects
    list_objs = measure.regionprops(img_segmentation_all_nuclei)
    for region_i in list_objs:
        if region_i.area < th_obj:
            img_segmentation_all_nuclei[img_segmentation_all_nuclei == region_i.label] = 0 
    
    del list_objs, model_trained_all_nuclei
    
    model_trained_regular_nuclei = models.CellposeModel(pretrained_model=path_model_regular_nuclei, gpu=flag_gpu)
    img_segmentation_regular_nuclei, flows, styles = model_trained_regular_nuclei.eval(img_nuclei_original, diameter=None, channels= channels)
    
    #Remove small objects
    list_objs = measure.regionprops(img_segmentation_regular_nuclei)
    for region_i in list_objs:
        if region_i.area < th_obj:
            img_segmentation_regular_nuclei[img_segmentation_regular_nuclei == region_i.label] = 0 
    
    del list_objs, model_trained_regular_nuclei
    
    model_trained_mitotic_nuclei = models.CellposeModel(pretrained_model=path_model_mitotic_nuclei, gpu=flag_gpu)
    img_segmentation_mitotic_nuclei, flows, styles = model_trained_mitotic_nuclei.eval(img_nuclei_original, diameter=None, channels= channels)
    
    del model_trained_mitotic_nuclei
    
    model_trained_xy_mitotic = models.CellposeModel(pretrained_model=path_model_xy_mitotic, gpu=flag_gpu)
    img_segmentation_xy_mitotic, flows, styles = model_trained_xy_mitotic.eval(img_nuclei_original, diameter=None, channels= channels)
    
    del model_trained_xy_mitotic
    
    model_trained_tubulin_mitosis = models.CellposeModel(pretrained_model=path_model_tubulin_mitosis, gpu=flag_gpu)
    img_segmentation_tubulin_mitosis, flows, styles = model_trained_tubulin_mitosis.eval(img_tubulin_original, diameter=None, channels= channels)
    
    list_objs = measure.regionprops(img_segmentation_tubulin_mitosis)
    for region_i in list_objs:
        if region_i.area < th_obj:
            img_segmentation_tubulin_mitosis[img_segmentation_tubulin_mitosis == region_i.label] = 0 
    
    del list_objs, model_trained_tubulin_mitosis
    
    
    # All nuclei AND regular nuclei AND no a-tubulin mitotic
    img_segmentation_regular_nuclei = get_overlapping_segmentation(img_segmentation_all_nuclei, img_segmentation_regular_nuclei, min_pixels=min_area_overlap, percentage = 0)
    
    img_segmentation_regular_nuclei = remove_overlapping_segmentation(img_segmentation_regular_nuclei, img_segmentation_tubulin_mitosis, min_pixels=min_area_overlap, percentage = percentage_overlap_nuclei)
    
    cell_props_regular_nuclei = get_props_per_cell(img_segmentation_regular_nuclei)
    n_regular_nuclei = len(cell_props_regular_nuclei)
    

    img_vertices, labels_vertices, number_vertices, sub_images_vertices = get_tubulin_vertices(img_tubulin_original, img_segmentation_tubulin_mitosis)
    
    # (Mitotic Nuclei - Regular Nuclei) AND a-tubulin mitotic AND a-tubulin equal or less than two vertices
    img_segmentation_all_regular_mitotic_nuclei = remove_overlapping_segmentation(img_segmentation_mitotic_nuclei, img_segmentation_regular_nuclei>0, min_pixels=min_area_overlap, percentage = 0)
    img_segmentation_all_regular_mitotic_nuclei = get_overlapping_segmentation(img_segmentation_all_regular_mitotic_nuclei, img_segmentation_tubulin_mitosis>0, min_pixels=min_area_overlap, percentage = 0)
    img_segmentation_all_regular_mitotic_nuclei = get_overlapping_segmentation(img_segmentation_all_regular_mitotic_nuclei, img_vertices==2, min_pixels=min_area_overlap, percentage = 0) #<=2
    cell_props_all_regular_mitotic_nuclei = get_props_per_cell(img_segmentation_all_regular_mitotic_nuclei)
    n_all_regular_mitotic_nuclei = len(cell_props_all_regular_mitotic_nuclei)
    
    # XY And tubulin mitosis AND tubulin equal or more than 3 vertices
    img_segmentation_xy_mitotic = get_overlapping_segmentation(img_segmentation_xy_mitotic, img_segmentation_tubulin_mitosis>0, min_pixels=min_area_overlap, percentage = 0)
    img_segmentation_xy_mitotic = get_overlapping_segmentation(img_segmentation_xy_mitotic, img_vertices>=3, min_pixels=min_area_overlap, percentage = 0)
    cell_props_xy_mitotic = get_props_per_cell(img_segmentation_xy_mitotic)
    n_xy_mitotic = len(cell_props_xy_mitotic)
    
    img_segmentation_tubulin_mitosis_in_xy_mitotic = get_overlapping_segmentation(img_segmentation_tubulin_mitosis, img_segmentation_xy_mitotic>0, min_pixels=min_area_overlap, percentage = 0)
    
    # Regular mitotic final
    img_segmentation_all_regular_mitotic_nuclei_2 = remove_overlapping_segmentation(img_segmentation_all_regular_mitotic_nuclei, img_segmentation_xy_mitotic>0, min_pixels=min_area_overlap, percentage = 0)
    cell_props_all_regular_mitotic_nuclei_2 = get_props_per_cell(img_segmentation_all_regular_mitotic_nuclei_2)
    n_all_regular_mitotic_nuclei_2 = len(cell_props_all_regular_mitotic_nuclei_2)
    
    # Overlap a-tubulin segmentation and regular nuclei
    img_segmentation_tubulin_mitosis_in_regular_mitotic = get_overlapping_segmentation(img_segmentation_tubulin_mitosis, img_segmentation_all_regular_mitotic_nuclei_2, min_pixels=min_area_overlap, percentage = 0)
    
    n_groups_regular_mitosis = len(get_props_per_cell(img_segmentation_tubulin_mitosis_in_regular_mitotic))
    
    #Compose an image with the xy mitotic and the tubulin segmentation
    composed_img_segmentation_regular_mitotic_nuclei_tubulin = draw_roi_over_image(img_segmentation_all_regular_mitotic_nuclei_2, img_segmentation_tubulin_mitosis_in_regular_mitotic)
    
    n_anaphase = n_all_regular_mitotic_nuclei_2 - n_groups_regular_mitosis #how many groups are double
    n_metaphase = n_groups_regular_mitosis - n_anaphase
    
    # Remaining nuclei, need to put somewhere:
    img_segmentation_all_nuclei_remaining = remove_overlapping_segmentation(img_segmentation_all_nuclei, img_segmentation_regular_nuclei>0, min_pixels=min_area_overlap, percentage = 0)
    img_segmentation_all_nuclei_remaining = remove_overlapping_segmentation(img_segmentation_all_nuclei_remaining, img_segmentation_all_regular_mitotic_nuclei_2>0, min_pixels=min_area_overlap, percentage = 0)
    img_segmentation_all_nuclei_remaining = remove_overlapping_segmentation(img_segmentation_all_nuclei_remaining, img_segmentation_xy_mitotic>0, min_pixels=min_area_overlap, percentage = 0)
    
    # Remaining tubulin
    # Remove tubulin in xy mitosis
    img_segmentation_tubulin_remaining = remove_overlapping_segmentation(img_segmentation_tubulin_mitosis, img_segmentation_xy_mitotic>0, min_pixels=min_area_overlap, percentage = 0)
    # Remove tubulin in regular mitosis
    img_segmentation_tubulin_remaining = remove_overlapping_segmentation(img_segmentation_tubulin_remaining, img_segmentation_tubulin_mitosis_in_regular_mitotic>0, min_pixels=min_area_overlap, percentage = 0)
    # Remove tubulin with only one vertice
    img_segmentation_tubulin_remaining = remove_overlapping_segmentation(img_segmentation_tubulin_remaining, img_vertices<=1, min_pixels=min_area_overlap, percentage = 0)
    
    # Get only those where remaining tubulin matches cell
    img_segmentation_all_nuclei_remaining_mitotic = get_overlapping_segmentation(img_segmentation_all_nuclei_remaining, img_segmentation_tubulin_remaining, min_pixels=min_area_overlap, percentage = 0)
    img_segmentation_all_nuclei_remaining_no_mitotic = remove_overlapping_segmentation(img_segmentation_all_nuclei_remaining, img_segmentation_all_nuclei_remaining_mitotic, min_pixels=min_area_overlap, percentage = 0)
    
    # Those mitotic not captured by the networks, have 2 vertices or more?
    img_segmentation_tubulin_remaining_with_nuclei_confirmed = get_overlapping_segmentation(img_segmentation_tubulin_remaining, img_segmentation_all_nuclei_remaining_mitotic, min_pixels=min_area_overlap, percentage = 0)
    img_vertices_tubulin_remaining_with_nuclei_confirmed = get_overlapping_segmentation(img_vertices, img_segmentation_tubulin_remaining_with_nuclei_confirmed, min_pixels=min_area_overlap, percentage = 0)
    _ , _ , list_nuclei_tubulin = matching_label_pairs(img_segmentation_all_nuclei_remaining_mitotic, img_segmentation_tubulin_remaining_with_nuclei_confirmed, min_pixels=min_area_overlap)
    list_tubulin_remaining = np.unique(img_segmentation_tubulin_remaining_with_nuclei_confirmed)
    list_tubulin_remaining = list_tubulin_remaining[list_tubulin_remaining>0] #No background
    
    img_segmentation_all_regular_mitotic_nuclei_3 = np.copy(img_segmentation_all_regular_mitotic_nuclei_2)
    img_segmentation_tubulin_mitosis_in_regular_mitotic_3 = np.copy(img_segmentation_tubulin_mitosis_in_regular_mitotic)
    img_segmentation_xy_mitotic_3 = np.copy(img_segmentation_xy_mitotic)
    img_segmentation_tubulin_mitosis_in_xy_mitotic_3 = np.copy(img_segmentation_tubulin_mitosis_in_xy_mitotic)
    img_segmentation_regular_nuclei_3 = np.copy(img_segmentation_regular_nuclei)
    
    for label_tubulin in list_tubulin_remaining:
        number_vertices_of_tubulin = np.median(img_vertices_tubulin_remaining_with_nuclei_confirmed[img_segmentation_tubulin_remaining_with_nuclei_confirmed==label_tubulin])
        list_nuclei_in_tubulin = []
        for pair_nuclei_tubulin in list_nuclei_tubulin:
            if pair_nuclei_tubulin[1] == label_tubulin:
                list_nuclei_in_tubulin.append(pair_nuclei_tubulin[0])
        
        if number_vertices_of_tubulin <=2:
            #Move to regular mitotic
            for label_nuclei in list_nuclei_in_tubulin:
                if label_nuclei>0:
                    
                    max_label = np.max(img_segmentation_all_regular_mitotic_nuclei_3)
                    img_segmentation_all_regular_mitotic_nuclei_3 = np.where(img_segmentation_all_nuclei_remaining==label_nuclei,max_label + 1,img_segmentation_all_regular_mitotic_nuclei_3)
                    
                    img_segmentation_tubulin_mitosis_in_regular_mitotic_3 = np.where(img_segmentation_tubulin_remaining_with_nuclei_confirmed==label_tubulin,label_nuclei,img_segmentation_tubulin_mitosis_in_regular_mitotic_3)
        else:
            #Move to XY mitotic
            for label_nuclei in list_nuclei_in_tubulin:
                if label_nuclei>0:
                    
                    max_label = np.max(img_segmentation_xy_mitotic_3)
                    img_segmentation_xy_mitotic_3 = np.where(img_segmentation_all_nuclei_remaining==label_nuclei,max_label + 1,img_segmentation_xy_mitotic_3)
                    
                    img_segmentation_tubulin_mitosis_in_xy_mitotic_3 = np.where(img_segmentation_tubulin_remaining_with_nuclei_confirmed==label_tubulin, label_tubulin,img_segmentation_tubulin_mitosis_in_xy_mitotic_3)
    
    # 
    list_all_nuclei_remaining_no_mitotic = np.unique(img_segmentation_all_nuclei_remaining_no_mitotic)
    list_all_nuclei_remaining_no_mitotic = list_all_nuclei_remaining_no_mitotic[list_all_nuclei_remaining_no_mitotic>0] #No background
    for label_nuclei_remaining in list_all_nuclei_remaining_no_mitotic:
        max_label = np.max(img_segmentation_regular_nuclei_3)
        img_segmentation_regular_nuclei_3 = np.where(img_segmentation_all_nuclei_remaining_no_mitotic==label_nuclei_remaining,max_label + 1,img_segmentation_regular_nuclei_3)
    
    #Final result
    
    composed_img_segmentation_regular_nuclei_final = draw_roi_over_image(img_nuclei_original, img_segmentation_regular_nuclei_3)
    n_regular_nuclei_final = len(get_props_per_cell(img_segmentation_regular_nuclei_3))
    composed_img_segmentation_regular_mitotic_nuclei_final = draw_roi_over_image(img_nuclei_original, img_segmentation_tubulin_mitosis_in_regular_mitotic_3)
    n_all_regular_mitotic_nuclei_3 = len(get_props_per_cell(img_segmentation_all_regular_mitotic_nuclei_3))
    n_groups_regular_mitosis_final = len(get_props_per_cell(img_segmentation_tubulin_mitosis_in_regular_mitotic_3))
    n_anaphase_final = n_all_regular_mitotic_nuclei_3 - n_groups_regular_mitosis_final #how many groups are double
    n_metaphase_final = n_groups_regular_mitosis_final - n_anaphase_final
    
    composed_img_segmentation_xy_final = draw_roi_over_image(img_nuclei_original, img_segmentation_tubulin_mitosis_in_xy_mitotic_3)
    n_xy_mitotic_final = len(get_props_per_cell(img_segmentation_tubulin_mitosis_in_xy_mitotic_3))
    
    composed_img_segmentation_tubulin_mitosis_regular_final = draw_roi_over_image(img_tubulin_original,img_segmentation_tubulin_mitosis_in_regular_mitotic_3)
    composed_img_segmentation_tubulin_mitosis_xy_final = draw_roi_over_image(img_tubulin_original,img_segmentation_tubulin_mitosis_in_xy_mitotic_3)
    
    
    plt.rcParams["figure.figsize"] = [20.50, 10.50]
    plt.rcParams["figure.autolayout"] = True
    
    fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
    
    plt.subplot(2,3, 1)
    plt.imshow(img_nuclei_original,cmap='gray')
    plt.gca().set_title('Original DAPI')
    
    plt.subplot(2,3, 4)
    plt.imshow(composed_img_segmentation_regular_nuclei_final)
    
    plt.gca().set_title('Regular nuclei: '+ str(n_regular_nuclei_final))
    
    plt.subplot(2,3, 2)
    plt.imshow(composed_img_segmentation_regular_mitotic_nuclei_final)
    
    plt.gca().set_title( 'Metaphase: ' + str(n_metaphase_final) + ' - '  + 'Anaphase: ' + str(n_anaphase_final))
    
    plt.subplot(2,3, 5)
    plt.imshow(composed_img_segmentation_xy_final)
    plt.gca().set_title('X-Y: ' + str(n_xy_mitotic_final))
    
    plt.subplot(2,3, 3)
    plt.imshow(composed_img_segmentation_tubulin_mitosis_regular_final)
    plt.gca().set_title('a-tubulin mitosis regular')
    
    plt.subplot(2,3, 6)
    plt.imshow(composed_img_segmentation_tubulin_mitosis_xy_final)
    plt.gca().set_title('a-tubulin mitosis XY')
    
    figManager = plt.get_current_fig_manager()
        
    
    plt.savefig(os.path.join(folder_output, sample_name + '_segmentations.png'), dpi=400)
    
    _01 = os.path.join(folder_output, sample_name + '_01_segmentation_regular_nuclei_'      + str(n_regular_nuclei_final) + '.png')
    _02 = os.path.join(folder_output, sample_name + '_02_segmentation_regular_mitotic_Metaphase_' + str(n_metaphase_final) + '_Anaphase_' + str(n_anaphase_final) + '.png')
    _03 = os.path.join(folder_output, sample_name + '_03_segmentation_xy_mitotic_'          + str(n_xy_mitotic_final)   + '.png')
    _04 = os.path.join(folder_output, sample_name + '_04_segmentation_tubulin_mitosis_regular.png')
    _05 = os.path.join(folder_output, sample_name + '_04_segmentation_tubulin_xy_mitosis_.png')
    
    cv2.imwrite(_01, cv2.cvtColor(composed_img_segmentation_regular_nuclei_final, cv2.COLOR_RGB2BGR))
    cv2.imwrite(_02, cv2.cvtColor(composed_img_segmentation_regular_mitotic_nuclei_final, cv2.COLOR_RGB2BGR))
    cv2.imwrite(_03, cv2.cvtColor(composed_img_segmentation_xy_final, cv2.COLOR_RGB2BGR))
    cv2.imwrite(_04, cv2.cvtColor(composed_img_segmentation_tubulin_mitosis_regular_final, cv2.COLOR_RGB2BGR))
    cv2.imwrite(_05, cv2.cvtColor(composed_img_segmentation_tubulin_mitosis_xy_final, cv2.COLOR_RGB2BGR))
    
    del composed_img_segmentation_regular_nuclei_final, composed_img_segmentation_regular_mitotic_nuclei_final, composed_img_segmentation_xy_final
    del composed_img_segmentation_tubulin_mitosis_regular_final, composed_img_segmentation_tubulin_mitosis_xy_final
    del img_segmentation_all_regular_mitotic_nuclei_3, img_segmentation_tubulin_mitosis_in_regular_mitotic_3, img_segmentation_xy_mitotic_3
    del img_segmentation_tubulin_mitosis_in_xy_mitotic_3, img_segmentation_regular_nuclei_3
    del img_nuclei_original, img_tubulin_original
    
    plt.close('all')
     
