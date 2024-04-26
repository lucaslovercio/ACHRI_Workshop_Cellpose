#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:40:04 2024

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
from quantify_segmentation import get_density_bins
from scipy import stats
from scipy import ndimage

def vertical_center_of_mass(subimage):
    total_intensity = 0
    total_weighted_intensity = 0
    
    for y, row in enumerate(subimage):
        for x, pixel in enumerate(row):
            total_intensity += pixel
            total_weighted_intensity += pixel * y
    
    if total_intensity == 0:
        return None
    else:
        return total_weighted_intensity / total_intensity


def vertical_standard_deviation(subimage, center_of_mass):
    if center_of_mass is None:
        return None

    total_intensity = 0
    total_weighted_squared_deviation = 0

    for y, row in enumerate(subimage):
        for x, pixel in enumerate(row):
            total_intensity += pixel
            total_weighted_squared_deviation += pixel * (y - center_of_mass) ** 2

    if total_intensity == 0:
        return None
    else:
        variance = total_weighted_squared_deviation / total_intensity
        return np.sqrt(variance)
    
def detect_transition_mask(mask, shift = 1):
    # Shift the mask downwards by one row and perform a bitwise XOR operation
    shifted_mask = np.roll(mask, shift, axis=0)
    transition_mask = (mask ^ shifted_mask) & mask
    return transition_mask


def create_mask_from_center_of_mass(subimage, center_of_mass, std_deviation):
    if center_of_mass is None or std_deviation is None:
        mask = np.zeros_like(subimage)
        return mask

    mask = np.zeros_like(subimage)
    center_y = int(round(center_of_mass))

    upper_limit = max(0, center_y - int(2 * std_deviation))
    lower_limit = min(len(subimage), center_y + int(2 * std_deviation))

    mask[upper_limit:lower_limit, :] = 1

    return mask

def combine_submasks(masks):
    # Initialize combined mask with zeros
    combined_mask = masks[0]

    # Combine masks
    for mask in masks[1:]:
        #print(combined_mask.shape)
        combined_mask = np.concatenate((combined_mask, mask), axis=1)
        
    return combined_mask

def crop_horizontal(image, subimage_width):
    cropped_images = []
    image_width = len(image[0])

    for start in range(0, image_width, subimage_width):
        end = min(start + subimage_width, image_width)
        cropped_images.append([row[start:end] for row in image])

    return cropped_images


def get_top_cell_label(labels, margin = 3, previous_label = -1):
    h = len(labels)
    w = len(labels[0])
    
    for i in range(h):
        row = labels[i]
        for j in range(margin,w-margin):
            if row[j] > 0 and row[j] != previous_label: #Is not background and not the previous detected
                return row[j], i, j
    
    return None, None, None  # No cell found in the subimage

def get_bottom_cell_label(labels, margin = 3, previous_label = -1):
    h = len(labels)
    w = len(labels[0])
    
    for i in reversed(range(h)):
        row = labels[i]
        for j in range(margin,w-margin):
            if row[j] > 0 and row[j] != previous_label: #Is not background and not the previous detected
                return row[j], i, j
    
    return None  # No cell found in the subimage

def get_mask_center_of_mass(nuclei_segmentation, subimage_width = 100):
    binary_segmentation = (nuclei_segmentation>0).astype(int)
    subimages = crop_horizontal(binary_segmentation, subimage_width)
    centers_mass = []
    std_mass = []
    masks = []
    
    for subimage in subimages:
        center_of_mass = vertical_center_of_mass(subimage)
        std_deviation = vertical_standard_deviation(subimage, center_of_mass)
        #print("Center of mass:", center_of_mass, "Vertical Standard Deviation:", std_deviation)
        centers_mass.append(center_of_mass)
        std_mass.append(std_deviation)
        mask = create_mask_from_center_of_mass(subimage, center_of_mass, std_deviation)
        #print(mask.shape)
        masks.append(mask)
    
    combined_mask = combine_submasks(masks)
    return combined_mask

def get_top_cells_labels(nuclei_segmentation, subimage_width = 100, bottom_cells = False):
    cell_labels = []
    cells_xy = []
    subimages = crop_horizontal(nuclei_segmentation, subimage_width)
    i_subimage = 0
    previous_label = -1 #To avoid detect twice the same cell in between two subimages
    for subimage in subimages:
        # Extracting label of the top cell in the subimage
        if not bottom_cells:
            top_label, y, x = get_top_cell_label(subimage, previous_label = previous_label)
        else:
            top_label, y, x = get_bottom_cell_label(subimage, previous_label = previous_label)
        if top_label is not None:
            previous_label = top_label
            #Correct x pos
            x = x + (i_subimage * subimage_width)
            cell_labels.append(top_label)
            cells_xy.append([x,y])
        i_subimage = i_subimage + 1
    
    return cell_labels, cells_xy, 

def get_layer_nuclei_histogram(nuclei_segmentation, vector_bins, min_cells_bin = 2, flag_show = False):
    dims = nuclei_segmentation.shape #dimY, dimX
    total_bins = len(vector_bins)
    
    #Get bins that are together
    vector_bins_th = vector_bins >= min_cells_bin
    labeled_array, num_features = ndimage.label(vector_bins_th)
    
    component_sizes = ndimage.sum(vector_bins_th, labeled_array, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1
    
    largest_component_indices = np.where(labeled_array == largest_component_label)
    start = min(largest_component_indices[0])
    end = max(largest_component_indices[0])
    
    
    start_row = np.int16(np.floor(dims[0] / total_bins) * start)
    end_row = np.int16((dims[0] / total_bins) * (end + 1))
    
    mask = np.zeros_like(nuclei_segmentation)
    mask[start_row:end_row, :] = 1
    
    not_outlier_nuclei = np.where(mask,nuclei_segmentation,0)
    return not_outlier_nuclei

def get_layer_nuclei_center_of_mass(nuclei_segmentation, subimage_width = 100, flag_show = False):
        
    combined_mask = get_mask_center_of_mass(nuclei_segmentation, subimage_width = subimage_width)
    
    if flag_show:
        plt.figure()
        plt.imshow(combined_mask)
        plt.title('combined_mask')
        plt.axis('off')
        #plt.show()
    
    #Deletion of outliers
    not_outlier_nuclei = np.where(combined_mask,nuclei_segmentation,0)
    
    if flag_show:
        plt.figure()
        plt.imshow(not_outlier_nuclei,cmap='gist_ncar')
        plt.gca().set_title('C3 Seg combined_mask')
    
    #Redo mask
    combined_mask = get_mask_center_of_mass(not_outlier_nuclei, subimage_width = subimage_width)
    top_edge = detect_transition_mask(combined_mask)
    
    if flag_show:
        plt.figure()
        plt.imshow(top_edge)
        plt.title('top_edge')
        plt.axis('off')
        #plt.show()
    
    layer_nuclei = np.where(combined_mask,nuclei_segmentation,0)
    
    return layer_nuclei
    
    
def plot_nuclei_segmentations(C1_original, C2_original, C3_original, C4_original,\
                              C1_segmentation, C2_segmentation, C3_segmentation, C4_segmentation,\
                                  C2_segmentation_match_nuclei, C3_segmentation_match_nuclei, C4_segmentation_match_nuclei,\
                                      path_to_save, mask_nuclei = None):
        
    plt.rcParams["figure.autolayout"] = True
    
    fig, ax = plt.subplots(2, 6, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
    
    plt.subplot(2, 6, 1)
    plt.imshow(C1_original,cmap='gray')
    #plt.gca().set_aspect('equal')
    #plt.gca().axis('equal')
    plt.gca().set_title('C1')
    
    if mask_nuclei is not None:
        plt.subplot(2, 6, 2)
        plt.imshow(mask_nuclei,cmap='gray')
        #plt.gca().set_aspect('equal')
        #plt.gca().axis('equal')
        plt.gca().set_title('C1 - mask for Cs')
    
    plt.subplot(2, 6, 3)
    plt.imshow(C1_segmentation,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    #plt.gca().axis('equal')
    plt.gca().set_title('C1 Seg')
    
    #Channel 2
    
    plt.subplot(2, 6, 4)
    plt.imshow(C2_original,cmap='gray')
    #plt.gca().set_aspect('equal')
    #plt.gca().axis('equal')
    plt.gca().set_title('C2')
    
    plt.subplot(2, 6, 5)
    plt.imshow(C2_segmentation,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    #plt.gca().axis('equal')
    plt.gca().set_title('C2 Seg')
    
    plt.subplot(2, 6, 6)    
    plt.imshow(C2_segmentation_match_nuclei,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    #plt.gca().axis('equal')
    plt.gca().set_title('C2 Seg - masked')
    
    #Channel 3
    
    plt.subplot(2, 6, 7)
    plt.imshow(C3_original,cmap='gray')
    #plt.gca().set_aspect('equal')
    plt.gca().set_title('C3')
    
    plt.subplot(2, 6, 8)
    plt.imshow(C3_segmentation,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    plt.gca().set_title('C3 Seg')
    
    plt.subplot(2, 6, 9)
    plt.imshow(C3_segmentation_match_nuclei,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    plt.gca().set_title('C3 Seg - masked')
    
    #Channel 4
    
    plt.subplot(2, 6, 10)
    plt.imshow(C4_original,cmap='gray')
    #plt.gca().set_aspect('equal')
    plt.gca().set_title('C4')
    
    plt.subplot(2, 6, 11)
    plt.imshow(C4_segmentation,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    plt.gca().set_title('C4 Seg')
    
    plt.subplot(2, 6, 12)
    plt.imshow(C4_segmentation_match_nuclei,cmap='gist_ncar')
    #plt.gca().set_aspect('equal')
    plt.gca().set_title('C4 Seg - Masked')
    
    figManager = plt.get_current_fig_manager()
        
    #figManager.frame.Maximize(True)    
    #figManager.window.showMaximized()
    
    #plt.figure(figsize=(8, 6))
    #plt.show()
    
    # Save the plot to a PNG file
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)
        
    return fig, ax
    
def get_distribution_histograms(cell_props_C1, cell_props_C2, cell_props_C3, cell_props_C4, dims, n_bins=20, path_to_save = None):
    # HISTOGRAM
    fig, ax = plt.subplots(2,2, figsize=(12, 6), 
                           constrained_layout = True)

    count_C1, edges_C1 = get_density_bins(cell_props_C1, dims[1],dims[0], axis=1, n_bins=n_bins)
    count_C2, edges_C2 = get_density_bins(cell_props_C2, dims[1],dims[0], axis=1, n_bins=n_bins)
    count_C3, edges_C3 = get_density_bins(cell_props_C3, dims[1],dims[0], axis=1, n_bins=n_bins)
    count_C4, edges_C4 = get_density_bins(cell_props_C4, dims[1],dims[0], axis=1, n_bins=n_bins)

    #Plot with the max value of any bin of the nuclei segmentation
    max_value_bin = np.max(count_C1)

    plt.subplot(2, 2, 1)
    labels_x = [str(x) for x in edges_C1[:-1]]
    plt.bar(labels_x,count_C1)
    plt.ylim(0, max_value_bin)
    plt.gca().set_title('C1')


    plt.subplot(2, 2, 2)
    labels_x = [str(x) for x in edges_C2[:-1]]
    plt.bar(labels_x,count_C2)
    plt.ylim(0, max_value_bin)
    plt.gca().set_title('C2')

    plt.subplot(2, 2, 3)
    labels_x = [str(x) for x in edges_C3[:-1]]
    plt.bar(labels_x,count_C3)
    plt.ylim(0, max_value_bin)
    plt.gca().set_title('C3')

    plt.subplot(2, 2, 4)
    labels_x = [str(x) for x in edges_C4[:-1]]
    plt.bar(labels_x,count_C4)
    plt.ylim(0, max_value_bin)
    plt.gca().set_title('C4')

    figManager = plt.get_current_fig_manager()
    
    #mng = plt.get_current_fig_manager()
    #figManager.frame.Maximize(True)    
    #figManager.window.showMaximized()
    #plt.figure(figsize=(8, 6))
    #plt.show()

    # Save the plot to a PNG file
    #plt.savefig(os.path.join(folder_output, sample_name + '_histograms.png'), dpi=300)
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)
        
    return count_C1, count_C2, count_C3, count_C4

def fit_cells(cells_xy):
    x = [point[0] for point in cells_xy]
    y = [point[1] for point in cells_xy]
    
    # Fit linear regression model
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return slope, intercept, r_value, p_value, std_err

def plot_cells(ax_subplot, cells_xy, color='red', marker_size = None):
    
    x = [point[0] for point in cells_xy]
    y = [point[1] for point in cells_xy]

    ax_subplot.scatter(x, y, color='red', marker='o', s=marker_size)
    
# TODO:: Need to refactor:
    
def get_different_fittings_center_of_mass(numpydata_C1_segmentation, numpydata_C2_segmentation_match_nuclei, numpydata_C3_segmentation_match_nuclei, numpydata_C4_segmentation_match_nuclei,\
                           d_start = 50, d_step = 10, d_end = 120):

    subimage_widths = np.arange(d_start, d_end+1, d_step)
    C2_list_std_err = []
    C3_list_std_err = []
    C4_list_std_err = []
    C1_list_std_err = []
    
    C2_list_r_value = []
    C3_list_r_value = []
    C4_list_r_value = []
    C1_list_r_value = []
    
    for subimage_width in subimage_widths:
        #print('subimage_width: ' + str(subimage_width))
        #C1_layer_nuclei = get_layer_nuclei(numpydata_C1_segmentation, subimage_width = subimage_width, flag_show = False)
        C2_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C2_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C3_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C3_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C4_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C4_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C1_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C1_segmentation, subimage_width = subimage_width, flag_show = False)
        
        C1_top_cell_labels, C1_top_cells_xy = get_top_cells_labels(C1_layer_nuclei, subimage_width = subimage_width)        
        C3_top_cell_labels, C3_top_cells_xy = get_top_cells_labels(C3_layer_nuclei, subimage_width = subimage_width)
        C4_top_cell_labels, C4_top_cells_xy = get_top_cells_labels(C4_layer_nuclei, subimage_width = subimage_width)
        C2_top_cell_labels, C2_top_cells_xy = get_top_cells_labels(C2_layer_nuclei, subimage_width = subimage_width)

        C1_top_slope, C1_top_intercept, C1_top_r_value, C1_top_p_value, C1_top_std_err = fit_cells(C1_top_cells_xy)        
        C2_top_slope, C2_top_intercept, C2_top_r_value, C2_top_p_value, C2_top_std_err = fit_cells(C2_top_cells_xy)
        C3_top_slope, C3_top_intercept, C3_top_r_value, C3_top_p_value, C3_top_std_err = fit_cells(C3_top_cells_xy)
        C4_top_slope, C4_top_intercept, C4_top_r_value, C4_top_p_value, C4_top_std_err = fit_cells(C4_top_cells_xy)

        C1_list_std_err.append(C1_top_std_err)        
        C2_list_std_err.append(C2_top_std_err)
        C3_list_std_err.append(C3_top_std_err)
        C4_list_std_err.append(C4_top_std_err)

        C1_list_r_value.append(C1_top_r_value)        
        C2_list_r_value.append(C2_top_r_value)
        C3_list_r_value.append(C3_top_r_value)
        C4_list_r_value.append(C4_top_r_value)
        
    return subimage_widths, C1_list_std_err, C2_list_std_err, C3_list_std_err, C4_list_std_err, C1_list_r_value, C2_list_r_value, C3_list_r_value, C4_list_r_value

def get_different_fittings_histogram(numpydata_C1_segmentation, numpydata_C2_segmentation_match_nuclei, numpydata_C3_segmentation_match_nuclei, numpydata_C4_segmentation_match_nuclei,
                                     C1_bins, C2_bins, C3_bins, C4_bins,\
                                         d_start = 50, d_step = 10, d_end = 120):

    subimage_widths = np.arange(d_start, d_end+1, d_step)
    C1_list_std_err = []
    C2_list_std_err = []
    C3_list_std_err = []
    C4_list_std_err = []

    C1_list_r_value = []    
    C2_list_r_value = []
    C3_list_r_value = []
    C4_list_r_value = []
    
    for subimage_width in subimage_widths:
        #print('subimage_width: ' + str(subimage_width))
        #C1_layer_nuclei = get_layer_nuclei(numpydata_C1_segmentation, subimage_width = subimage_width, flag_show = False)
        C1_layer_nuclei = get_layer_nuclei_histogram(numpydata_C1_segmentation, C1_bins)
        C2_layer_nuclei = get_layer_nuclei_histogram(numpydata_C2_segmentation_match_nuclei, C2_bins)
        C3_layer_nuclei = get_layer_nuclei_histogram(numpydata_C3_segmentation_match_nuclei, C3_bins)
        C4_layer_nuclei = get_layer_nuclei_histogram(numpydata_C4_segmentation_match_nuclei, C4_bins)

        C1_top_cell_labels, C1_top_cells_xy = get_top_cells_labels(C1_layer_nuclei, subimage_width = subimage_width)    
        C3_top_cell_labels, C3_top_cells_xy = get_top_cells_labels(C3_layer_nuclei, subimage_width = subimage_width)
        C4_top_cell_labels, C4_top_cells_xy = get_top_cells_labels(C4_layer_nuclei, subimage_width = subimage_width)
        C2_top_cell_labels, C2_top_cells_xy = get_top_cells_labels(C2_layer_nuclei, subimage_width = subimage_width)

        C1_top_slope, C1_top_intercept, C1_top_r_value, C1_top_p_value, C1_top_std_err = fit_cells(C1_top_cells_xy)        
        C2_top_slope, C2_top_intercept, C2_top_r_value, C2_top_p_value, C2_top_std_err = fit_cells(C2_top_cells_xy)
        C3_top_slope, C3_top_intercept, C3_top_r_value, C3_top_p_value, C3_top_std_err = fit_cells(C3_top_cells_xy)
        C4_top_slope, C4_top_intercept, C4_top_r_value, C4_top_p_value, C4_top_std_err = fit_cells(C4_top_cells_xy)

        C1_list_std_err.append(C1_top_std_err)        
        C2_list_std_err.append(C2_top_std_err)
        C3_list_std_err.append(C3_top_std_err)
        C4_list_std_err.append(C4_top_std_err)

        C1_list_r_value.append(C1_top_r_value)        
        C2_list_r_value.append(C2_top_r_value)
        C3_list_r_value.append(C3_top_r_value)
        C4_list_r_value.append(C4_top_r_value)
        
    return subimage_widths, C1_list_std_err, C2_list_std_err, C3_list_std_err, C4_list_std_err, C1_list_r_value, C2_list_r_value, C3_list_r_value, C4_list_r_value

    


