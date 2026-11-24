#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:40:04 2024

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
from quantify_segmentation import get_density_bins, get_joint_expr_per_cell, filter_joint_cell_expr, get_props_per_cell, get_labels, get_expr_from_labels
from quantify_segmentation import plot_expressions as plot_expressions_two_lists
from scipy import stats
from scipy import ndimage
from skimage import measure
import math
import os
import pandas as pd
import cv2

def plot_expressions(list_cell_expr1_expr2, title_plot, label_x = 'Channel 1', label_y = 'Channel 2'):
    # Each element is (label, mean_exp1, mean_exp2, pred1, pred2): split the two expressions into two lists
    # and plot them with quantify_segmentation.plot_expressions
    list_expr1 = [mean_exp1 for label, mean_exp1, mean_exp2, pred1, pred2 in list_cell_expr1_expr2]
    list_expr2 = [mean_exp2 for label, mean_exp1, mean_exp2, pred1, pred2 in list_cell_expr1_expr2]
    plot_expressions_two_lists(list_expr1, list_expr2, title_plot, label_x = label_x, label_y = label_y)

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
        #return None

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
        if mask is not None:
            combined_mask = np.concatenate((combined_mask, mask), axis=1)
        else:
            print('tried to concatenate a None Mask')
        
    return combined_mask

def crop_horizontal(image, subimage_width):
    cropped_images = []
    image_width = len(image[0])

    for start in range(0, image_width, subimage_width):
        end_sub = min(start + subimage_width, image_width)
        # print('start:' + str(start) + ' end:' + str(end_sub))
        cropped_images.append([row[start:end_sub] for row in image])


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
    
    return None, None, None  # No cell found in the subimage

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
        
        
        if mask is not None:
            masks.append(mask)
        else:
            print('mask is None')
        
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

def get_layer_nuclei_histogram(nuclei_segmentation, vector_bins, min_cells_bin = 8, flag_show = False):
    dims = nuclei_segmentation.shape #dimY, dimX
    total_bins = len(vector_bins)
    # print('total_bins: ' + str(total_bins))
    #Get bins that are together
    vector_bins_th = vector_bins >= min_cells_bin
    labeled_array, num_features = ndimage.label(vector_bins_th)
    
    component_sizes = ndimage.sum(vector_bins_th, labeled_array, range(1, num_features + 1))
    if component_sizes is None or len(component_sizes) == 0:
        return nuclei_segmentation, float("nan"), float("nan")
    largest_component_label = np.argmax(component_sizes) + 1
    
    largest_component_indices = np.where(labeled_array == largest_component_label)
    start = min(largest_component_indices[0])
    end = max(largest_component_indices[0])
    
    
    start_row = np.int16(np.floor(dims[0] / total_bins) * start)
    end_row = np.int16((dims[0] / total_bins) * (end + 1))
    # print('dims[0]: ' + str(dims[0]))
    
    # print('start: ' + str(start))
    # print('start_row: ' + str(start_row))
    # print('end: ' + str(end))
    # print('end_row: ' + str(end_row))
    mask = np.zeros_like(nuclei_segmentation)
    mask[start_row:end_row, :] = 1
    
    not_outlier_nuclei = np.where(mask,nuclei_segmentation,0)
    return not_outlier_nuclei, start_row, end_row
    
def get_segmentation_filtered_layer(numpydata_segmentation, start_row, end_row):
    segmentation_filtered_layer = np.copy(numpydata_segmentation)
    if math.isnan(start_row) or math.isnan(end_row):
        return segmentation_filtered_layer
    segmentation_filtered_layer[0:start_row,:] = 0
    segmentation_filtered_layer[end_row:,:] = 0
    return segmentation_filtered_layer

def get_number_of_cells(binary_image, min_px_object = 16):
    # Perform connected component analysis
    labels = measure.label(binary_image)
    
    # Count the number of pixels in each object
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    # Remove background label (label 0)
    unique_labels = unique_labels[1:]
    label_counts = label_counts[1:]
    
    # Print the number of pixels in each object
    n_cells = 0
    for label, count in zip(unique_labels, label_counts):
        if count > min_px_object:
            n_cells = n_cells + 1
    return n_cells

def get_layer_nuclei_center_of_mass(nuclei_segmentation, subimage_width = 100, flag_show = False):
    print('subimage_width: ' + str(subimage_width))    
    combined_mask = get_mask_center_of_mass(nuclei_segmentation, subimage_width = subimage_width)
    
    if flag_show:
        plt.figure()
        plt.imshow(combined_mask)
        plt.title('combined_mask')
        plt.axis('off')
        plt.show()
    
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
        plt.show()
    
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
    
    # figManager = plt.get_current_fig_manager()
        
    # figManager.window.showMaximized()
    # figManager.resize(*figManager.window.maxsize())
    
    #plt.figure(figsize=(8, 6))
    plt.show()
    
    # Save the plot to a PNG file
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)
        
    return fig, ax


def plot_nuclei_segmentations_list(list_original, list_segmentation, list_segmentation_match_nuclei,\
                                    path_to_save, mask_nuclei = None, channel_labels = None):
    """Generic version of plot_nuclei_segmentations for a variable number of channels (max 4).

    list_original, list_segmentation and list_segmentation_match_nuclei must be lists indexed like
    C1, C2, C3, (C4). Channel 0 (C1) does not use list_segmentation_match_nuclei (pass None there);
    its middle slot shows mask_nuclei instead, same as the original fixed-channel function.
    """

    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(2, 6, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")

    n_channels = len(list_original)
    for i_channel in range(n_channels):
        base = i_channel * 3
        label = channel_labels[i_channel] if channel_labels else 'C' + str(i_channel + 1)

        plt.subplot(2, 6, base + 1)
        plt.imshow(list_original[i_channel], cmap='gray')
        plt.gca().set_title(label)

        if i_channel == 0:
            if mask_nuclei is not None:
                plt.subplot(2, 6, base + 2)
                plt.imshow(mask_nuclei, cmap='gray')
                plt.gca().set_title(label + ' - mask for Cs')

            plt.subplot(2, 6, base + 3)
            plt.imshow(list_segmentation[i_channel], cmap='gist_ncar')
            plt.gca().set_title(label + ' Seg')
        else:
            plt.subplot(2, 6, base + 2)
            plt.imshow(list_segmentation[i_channel], cmap='gist_ncar')
            plt.gca().set_title(label + ' Seg')

            plt.subplot(2, 6, base + 3)
            plt.imshow(list_segmentation_match_nuclei[i_channel], cmap='gist_ncar')
            plt.gca().set_title(label + ' Seg - masked')

    plt.show()

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)

    return fig, ax


def get_distribution_histograms_list(list_cell_props, dims, n_bins=20, path_to_save = None, channel_labels = None):
    """Generic version of get_distribution_histograms for a variable number of channels (max 4)."""

    fig, ax = plt.subplots(2, 2, figsize=(12, 6),
                           constrained_layout = True)

    list_counts = []
    list_edges = []
    for cell_props in list_cell_props:
        count, edges = get_density_bins(cell_props, dims[1], dims[0], axis=1, n_bins=n_bins)
        list_counts.append(count)
        list_edges.append(edges)

    #Plot with the max value of any bin of the nuclei segmentation (C1)
    max_value_bin = np.max(list_counts[0])

    for i_channel in range(len(list_cell_props)):
        plt.subplot(2, 2, i_channel + 1)
        labels_x = [str(x) for x in list_edges[i_channel][:-1]]
        plt.bar(labels_x, list_counts[i_channel])
        plt.ylim(0, max_value_bin)
        plt.gca().set_title(channel_labels[i_channel] if channel_labels else 'C' + str(i_channel + 1))

    plt.show()

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)

    return list_counts

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

    # figManager = plt.get_current_fig_manager()
        
    # figManager.window.showMaximized()
    # figManager.resize(*figManager.window.maxsize())


    #plt.figure(figsize=(8, 6))
    plt.show()

    # Save the plot to a PNG file
    #plt.savefig(os.path.join(folder_output, sample_name + '_histograms.png'), dpi=300)
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)
        
    return count_C1, count_C2, count_C3, count_C4

def fit_cells(cells_xy):
    
    if cells_xy is None or len(cells_xy) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    
    x = [point[0] for point in cells_xy]
    y = [point[1] for point in cells_xy]
    
    # Fit linear regression model
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return slope, intercept, r_value, p_value, std_err

def plot_cells(ax_subplot, cells_xy, color='red', marker_size = None):
    
    x = [point[0] for point in cells_xy]
    y = [point[1] for point in cells_xy]

    ax_subplot.scatter(x, y, color='red', marker='o', s=marker_size)
    
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
        C1_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C1_segmentation, subimage_width = subimage_width, flag_show = False)
        C2_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C2_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C3_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C3_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C4_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C4_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
    
        C3_top_cell_labels, C3_top_cells_xy = get_top_cells_labels(C3_layer_nuclei, subimage_width = subimage_width)
        C4_top_cell_labels, C4_top_cells_xy = get_top_cells_labels(C4_layer_nuclei, subimage_width = subimage_width)
        C2_top_cell_labels, C2_top_cells_xy = get_top_cells_labels(C2_layer_nuclei, subimage_width = subimage_width)
        C1_top_cell_labels, C1_top_cells_xy = get_top_cells_labels(C1_layer_nuclei, subimage_width = subimage_width)
        
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

def get_different_fittings_histogram(C1_filtered, C2_filtered, C3_filtered, C4_filtered,\
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
        C1_top_cell_labels, C1_top_cells_xy = get_top_cells_labels(C1_filtered, subimage_width = subimage_width)
        C3_top_cell_labels, C3_top_cells_xy = get_top_cells_labels(C3_filtered, subimage_width = subimage_width)
        C4_top_cell_labels, C4_top_cells_xy = get_top_cells_labels(C4_filtered, subimage_width = subimage_width)
        C2_top_cell_labels, C2_top_cells_xy = get_top_cells_labels(C2_filtered, subimage_width = subimage_width)
        
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

def get_different_fittings_histogram_list(list_filtered, d_start = 50, d_step = 10, d_end = 120):
    """Generic version of get_different_fittings_histogram for a variable number of channels: for each subimage
    width, the std error and r value of the linear fitting of the top cells of each channel (list_filtered: the
    layer-filtered segmentations). Returns (subimage_widths, list_list_std_err, list_list_r_value), the lists
    indexed [channel][width]."""

    subimage_widths = np.arange(d_start, d_end+1, d_step)
    list_list_std_err = [[] for _ in list_filtered]
    list_list_r_value = [[] for _ in list_filtered]

    for subimage_width in subimage_widths:
        for i_channel, filtered in enumerate(list_filtered):
            _, top_cells_xy = get_top_cells_labels(filtered, subimage_width = subimage_width)
            _, _, top_r_value, _, top_std_err = fit_cells(top_cells_xy)
            list_list_std_err[i_channel].append(top_std_err)
            list_list_r_value[i_channel].append(top_r_value)

    return subimage_widths, list_list_std_err, list_list_r_value



def draw_dashed_hline(img, y, x_start, x_end, color, thickness, dash_length, gap_length):
    """Draws a dashed horizontal line (e.g. a layer limit top_y/bottom_y) on img, in place."""
    x = x_start
    while x < x_end:
        x_dash_end = min(x + dash_length, x_end)
        cv2.line(img, (x, y), (x_dash_end, y), color, thickness)
        x += dash_length + gap_length


def get_labels_in_y_range(cell_props, width, y_start, y_end, include_y_end=False):
    """Labels of the cells whose centroid Y falls in [y_start, y_end) - or [y_start, y_end] if
    include_y_end, for the last bin/half so that bottom_y itself is not left out (get_labels'
    bbox check is ymin <= yCentroid < ymax)."""
    bbox_y_end = y_end + 1 if include_y_end else y_end
    return get_labels(cell_props, bbox=(0, y_start, width, bbox_y_end))


def get_segmentation_with_labels(img_segmentation, labels):
    """Copy of img_segmentation keeping only the given labels (whole cells), the rest set to 0."""
    return np.where(np.isin(img_segmentation, labels), img_segmentation, 0)


def get_top_bottom_y(list_limits):
    """Y range [top_y, bottom_y] that covers the layers of list_limits ([start_row, end_row] of one or more layer
    channels): from the top-most start to the bottom-most end (end_row is exclusive, so bottom_y = max end - 1).
    NaN if any limit is NaN (a channel without layer)."""
    limits = np.concatenate([np.array(limits) for limits in list_limits])
    return np.min(limits), np.max(limits) - 1


def get_binned_expression_stats(img_segmentation, img_expression, channel_name, top_y, bottom_y, n_bins, flag_use_median=True):
    """Divides [top_y, bottom_y] into n_bins equal-height Y bins and, for each bin, computes the
    mean/median across cells of the per-cell expression (median pixel value inside that cell, or the
    mean if not flag_use_median - same convention as get_joint_expr_per_cell) for the cells whose
    centroid falls in that bin.
    img_segmentation is the labeled cell image used both for centroids (get_props_per_cell/
    get_labels) and for masking img_expression per cell (get_expr_from_labels). channel_name is
    only carried along as a column, so rows stay identifiable once samples get concatenated."""
    cell_props = get_props_per_cell(img_segmentation)
    width = img_segmentation.shape[1]
    bin_edges = np.linspace(top_y, bottom_y, n_bins + 1)

    rows = []
    for i_bin in range(n_bins):
        y_start, y_end = bin_edges[i_bin], bin_edges[i_bin + 1]
        labels_in_bin = get_labels_in_y_range(cell_props, width, y_start, y_end, include_y_end=(i_bin == n_bins - 1))
        expr_per_cell = get_expr_from_labels(labels_in_bin, img_segmentation, img_expression, flag_use_median = flag_use_median)

        rows.append({
            'channel': channel_name, 'bin': i_bin, 'y_start': y_start, 'y_end': y_end, 'n_cells': len(expr_per_cell),
            'mean_expr': np.mean(expr_per_cell) if expr_per_cell else np.nan,
            'median_expr': np.median(expr_per_cell) if expr_per_cell else np.nan,
        })
    return pd.DataFrame(rows)


def plot_binned_expression_boxplots(df_binned_all, folder_out, binned_rows):
    """Aggregation across samples: for each treatment and each binned expression row of binned_rows ((channel
    name, n_bins, layers name) triplets; layers name is the name(s) of the layer channel(s) whose layers give the
    Y range of the bins, joined with '_and_'), using df_binned_all (rows from get_binned_expression_stats plus
    'sample'/'treatment'/'layers' columns), one boxplot of the per-sample mean_expr and another of the per-sample
    median_expr, with one box per Y bin (bin 1 = top_y) - n_bins boxes, which can differ per row.
    The same is also done with all the samples together (every treatment), named 'all_samples'.
    Each point inside a box is one sample. Saves the figures plus the csv with the plotted data in folder_out."""
    # (name used in the files, name used in the title, rows): one group per treatment, plus all samples together
    groups = [(treatment, treatment, df_binned_all[df_binned_all['treatment'] == treatment])
              for treatment in sorted(df_binned_all['treatment'].unique())]
    groups.append(('all_samples', 'all samples', df_binned_all))

    for group_name, group_title, df_group in groups:
        for channel_name, n_bins, layers_name in binned_rows:
            df_sub = df_group[(df_group['layers'] == layers_name) & (df_group['channel'] == channel_name)]
            if df_sub.empty:
                continue

            bin_labels = [f'Bin {i + 1}' for i in range(n_bins)]
            bin_labels[0] += '\n(top)'
            bin_labels[-1] += '\n(bottom)'

            name_base = f'binned_expression_{group_name}_{channel_name}_in_{layers_name}'

            df_sub.to_csv(os.path.join(folder_out, name_base + '_Fig3d.csv'), index=False)

            for stat_col, stat_name in [('mean_expr', 'mean'), ('median_expr', 'median')]:
                # samples with no cells in a bin have NaN there - dropped from that box
                data = [df_sub.loc[df_sub['bin'] == i_bin, stat_col].dropna().values for i_bin in range(n_bins)]

                plt.figure(figsize=(8, 6))
                plt.boxplot(data, tick_labels=bin_labels)
                plt.ylim(bottom=0)
                plt.ylabel(f'{stat_name.capitalize()} of per-cell {channel_name} expression')
                plt.title(f'{channel_name} in the layers of {layers_name.replace("_and_", " and ")} - {group_title} ({df_sub["sample"].nunique()} samples)')
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(folder_out, name_base + '_' + stat_name + '_Fig3d_boxplot.png'), dpi=400)
                plt.close()


def keep_sample_table(dict_tables, path_csv, sample_name, df_table):
    """If dict_tables is not None, keeps df_table (the per-sample csv path_csv, whose file name is sample_name +
    suffix) as dict_tables[suffix], so that the same table of every sample can be aggregated across samples
    (script_analyze_layers.save_aggregated_tables)."""
    if dict_tables is not None:
        dict_tables[os.path.basename(path_csv)[len(sample_name):]] = df_table


def plot_intensity_ratio_histogram(list_cell_expr1_expr2, name_1, name_2, bin_min, bin_max, n_bins, sample_name, folder_output_expression, str_ending,
                                   dict_tables=None):
    """Fig 3F: given list_cell_expr1_expr2 ([label, expr_1, expr_2, pred1, pred2] per cell, e.g.
    list_cells_both_layers_expr1_expr2 with channel 1 = name_1 (e.g. Tbr1), channel 2 = name_2 (e.g. Bcl11b)),
    computes the per-cell log2(name_2/name_1) intensity ratio and saves it to csv plus an n_bins histogram
    over [bin_min, bin_max] - one sample at a time, samples get combined later (the csv is kept in dict_tables,
    see keep_sample_table)."""
    eps = 1e-7
    col_ratio = f'ratio_log2_{name_2}_{name_1}'
    rows = []
    for label, expr_1, expr_2, pred_1, pred_2 in list_cell_expr1_expr2:
        ratio_log2 = np.log2((expr_2 + eps) / (expr_1 + eps))
        rows.append({'label': label, f'expr_{name_1}': expr_1, f'expr_{name_2}': expr_2, col_ratio: ratio_log2})
    df_ratio = pd.DataFrame(rows, columns=['label', f'expr_{name_1}', f'expr_{name_2}', col_ratio])

    path_csv = os.path.join(folder_output_expression, sample_name + str_ending + '.csv')
    df_ratio.to_csv(path_csv, index=False)
    keep_sample_table(dict_tables, path_csv, sample_name, df_ratio)

    bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)
    plt.figure(figsize=(8, 8))
    plt.hist(df_ratio[col_ratio], bins=bin_edges)
    plt.xlabel(f'{name_2}/{name_1} (Log2, norm)')
    plt.ylabel('Density of cells')
    plt.title(sample_name)
    plt.savefig(os.path.join(folder_output_expression, sample_name + str_ending + '.png'), dpi=400)
    plt.close()

    return df_ratio


def plot_save_joint_expression(list_cell_expr1_expr2, name_1, name_2, title_suffix, path_base, flag_use_median=True,
                               sample_name=None, dict_tables=None):
    """Scatter of per-cell expression name_1 (X) vs name_2 (Y) (plot_expressions) saved as path_base + '.png',
    plus the same data saved as path_base + '.csv', with the channel names in the column headers (Median_Exp_
    or Mean_Exp_, as the per-cell expression was computed: flag_use_median of get_joint_expr_per_cell). The csv
    is kept in dict_tables (see keep_sample_table; path_base must then start with sample_name)."""
    plot_expressions(list_cell_expr1_expr2, title_plot = name_1 + ' vs ' + name_2 + title_suffix, label_x = name_1, label_y = name_2)
    plt.savefig(path_base + '.png', dpi=400)
    plt.close()
    stat_exp = 'Median_Exp_' if flag_use_median else 'Mean_Exp_'
    columns = ['Label', stat_exp + name_1, stat_exp + name_2, f'Pred_{name_1}', f'Pred_{name_2}']
    df_expr = pd.DataFrame(list_cell_expr1_expr2, columns=columns)
    df_expr.to_csv(path_base + '.csv', index=False)
    keep_sample_table(dict_tables, path_base + '.csv', sample_name, df_expr)


def get_joint_expr_per_half(img_segmentation, list_channels, list_channels_filtered, c_1, c_2, top_y, bottom_y, flag_use_median=True):
    """Splits [top_y, bottom_y] into a top and a bottom half (same as taking bins 1..n/2 and n/2+1..n
    of get_binned_expression_stats in the same range when n is even: 4 bins -> 1-2 / 3-4, 10 bins -> 1-5 / 6-10) and
    returns the per-cell joint expression (get_joint_expr_per_cell) of the cells whose centroid falls
    in each half: {'top_half': [...], 'bottom_half': [...]}."""
    cell_props = get_props_per_cell(img_segmentation)
    width = img_segmentation.shape[1]
    mid_y = (top_y + bottom_y) / 2

    halves = {'top_half': (top_y, mid_y, False), 'bottom_half': (mid_y, bottom_y, True)}
    dict_expr = {}
    for half_name, (y_start, y_end, include_y_end) in halves.items():
        labels_half = get_labels_in_y_range(cell_props, width, y_start, y_end, include_y_end=include_y_end)
        segmentation_half = get_segmentation_with_labels(img_segmentation, labels_half)
        dict_expr[half_name] = get_joint_expr_per_cell(segmentation_half, list_channels[c_1], list_channels[c_2],\
                                                       list_channels_filtered[c_1], list_channels_filtered[c_2],\
                                                       flag_use_median = flag_use_median)
    return dict_expr


def plot_generate_joint_expression_data(list_channels, list_channels_filtered, C1_segmentation_filtered_layer,\
                                         c_1, c_2, c_1_limits, c_2_limits, name_1, name_2, ending_norm_expression, sample_name, folder_output_expression,\
                                         ratio_bin_min, ratio_bin_max, ratio_n_bins,\
                                         dash_length_px, gap_length_px, flag_use_median=True, dict_tables=None):
    """Joint expression of channels c_1/c_2 (0-based indices into list_channels, named name_1/name_2,
    used for plot labels and file names) (Fig 3) for one sample: scatter plots + csvs (all cells, only
    positives, cells in both layers and its top/bottom halves), Fig 3F intensity ratio histogram and
    rgb debug images with the layer limits drawn as dashed lines. The expression of each cell is the median
    of its pixels, or their mean if not flag_use_median (scatters and Fig 3F ratio). The binned expression is
    computed apart (script_analyze_layers.analyze_binned_expression), with its own layers.
    The csvs of the scatters and of the Fig 3F ratio are also kept in dict_tables (keep_sample_table), to be
    aggregated across samples."""

    top_y, bottom_y = get_top_bottom_y([c_1_limits, c_2_limits])

    str_ending = '_expr_' + name_1 + '_' + name_2 + ending_norm_expression
    list_cell_expr1_expr2 = get_joint_expr_per_cell(C1_segmentation_filtered_layer, list_channels[c_1], list_channels[c_2],\
                                                        list_channels_filtered[c_1], list_channels_filtered[c_2],\
                                                        flag_use_median = flag_use_median)

    path_base = os.path.join(folder_output_expression, sample_name + str_ending)

    # all cells
    plot_save_joint_expression(list_cell_expr1_expr2, name_1, name_2, '', path_base + '_all_cells_in_image', flag_use_median = flag_use_median,
                               sample_name = sample_name, dict_tables = dict_tables)

    # only positives
    filtered_list_cell_expr1_expr2 = filter_joint_cell_expr(list_cell_expr1_expr2)
    del list_cell_expr1_expr2
    plot_save_joint_expression(filtered_list_cell_expr1_expr2, name_1, name_2, ' - Only positives', path_base + '_only_positives',
                               flag_use_median = flag_use_median, sample_name = sample_name, dict_tables = dict_tables)
    del filtered_list_cell_expr1_expr2

    # all in both layers
    cell_in_both_layers = get_segmentation_filtered_layer(C1_segmentation_filtered_layer, top_y, bottom_y)
    list_cells_both_layers_expr1_expr2 = get_joint_expr_per_cell(cell_in_both_layers, list_channels[c_1], list_channels[c_2],\
                                                            list_channels_filtered[c_1], list_channels_filtered[c_2],\
                                                            flag_use_median = flag_use_median)

    # Fig 3F: per-cell log2(name_2/name_1) intensity ratio histogram (e.g. Bcl11b/Tbr1)
    plot_intensity_ratio_histogram(list_cells_both_layers_expr1_expr2, name_1, name_2, ratio_bin_min, ratio_bin_max, ratio_n_bins,\
                                    sample_name, folder_output_expression, str_ending='_ratio_' + name_2 + '_' + name_1 + '_fig3f_partial',
                                    dict_tables = dict_tables)

    # all cells in both layers
    str_ending = '_expr_' + name_1 + '_' + name_2 + ending_norm_expression + '_cells_in_both_layers'
    path_base = os.path.join(folder_output_expression, sample_name + str_ending)
    plot_save_joint_expression(list_cells_both_layers_expr1_expr2, name_1, name_2, ' all cells in layers', path_base, flag_use_median = flag_use_median,
                               sample_name = sample_name, dict_tables = dict_tables)

    # cells in both layers, split into top half (bins 1..n/2) and bottom half (bins n/2+1..n)
    dict_expr_halves = get_joint_expr_per_half(cell_in_both_layers, list_channels, list_channels_filtered, c_1, c_2, top_y, bottom_y,
                                               flag_use_median = flag_use_median)
    for half_name, list_expr_half in dict_expr_halves.items():
        title_suffix = ' cells in layers, ' + half_name.replace('_', ' ')
        plot_save_joint_expression(list_expr_half, name_1, name_2, title_suffix, path_base + '_' + half_name + '_fig3k_partial',
                                   flag_use_median = flag_use_median, sample_name = sample_name, dict_tables = dict_tables)

    rgb_norm_channels = np.uint8(np.dstack([list_channels[c_2]*255, list_channels[c_1]*255, list_channels[0]*150])) # Nuclear in blue, usually they show BGR pictures
    cv2.imwrite(os.path.join(folder_output_expression, sample_name + str_ending + '_rgb.png'), cv2.cvtColor(rgb_norm_channels, cv2.COLOR_RGB2BGR))

    rgb_norm_channels_mag = np.uint8(np.dstack([list_channels[c_2]*255, list_channels[c_1]*255, np.zeros_like(list_channels[c_1]) ])) # Nuclear in blue, usually they show BGR pictures
    cv2.imwrite(os.path.join(folder_output_expression, sample_name + str_ending + '_rgb_mag.png'), cv2.cvtColor(rgb_norm_channels_mag, cv2.COLOR_RGB2BGR))

    rgb_with_lines = rgb_norm_channels.copy()
    width = rgb_with_lines.shape[1]
    height = rgb_with_lines.shape[0]
    print('width:',width,' height',height,' top_y',top_y,' bottom_y',bottom_y)
    if (not math.isnan(top_y)) and (not math.isnan(bottom_y) and (top_y < height) and (bottom_y < height)):
        draw_dashed_hline(rgb_with_lines, top_y, 0, width - 1, (255, 255, 255), 2, dash_length_px, gap_length_px)
        draw_dashed_hline(rgb_with_lines, bottom_y, 0, width - 1, (255, 255, 255), 2, dash_length_px, gap_length_px)
        cv2.imwrite(os.path.join(folder_output_expression, sample_name + str_ending + '_rgb_lines.png'), cv2.cvtColor(rgb_with_lines, cv2.COLOR_RGB2BGR))


    del list_cells_both_layers_expr1_expr2

