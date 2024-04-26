#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
###################################   PARAMETERS   #########################
############################################################################

#File path to .tif or .tiff file with a stitched stack of IHC images
tiff_path = '' #.tif file #In Windows, place an r before the ''

# Number of channels in tiff stack
n_channels = 4

#Number of bins for histogram
n_bins = 20

#Path to trained architectures
path_model_trained_C1  = ''#'Neurons_C1.183326' #In Windows, place an r before the ''
path_model_trained_C2  = ''#'Neurons_C2.919883' #In Windows, place an r before the ''
path_model_trained_C3  = ''#'Neurons_C3.981474' #In Windows, place an r before the ''
path_model_trained_C4  = ''#'Neurons_C4.909737' #In Windows, place an r before the ''


#Parameters for running the segmentation
flag_normalize = False
flag_gpu = False

#Width for compute nuclei in edges for the table edge_fitting
subimage_width = 100

#Side of the Square for morisita index for the table dispersion_indexes
d_morisita = 150

# Optional parameters for linear fitting of edge
flag_histbins_for_outliers = True
flag_text = True #If show the error of the linear fitting
text_shift = 60 # How much above the linear fitting the text can go
fitting_color = 'green' # Color of line and text
fontsize = 5
marker_size = 10

############################################################################
############################################################################
############################################################################


import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cellpose import models
from quantify_segmentation import get_props_per_cell
from aux_functions.functionPercNorm import functionPercNorm
from analyze_neuron_layers import get_layer_nuclei_histogram, get_layer_nuclei_center_of_mass, get_top_cells_labels,\
    get_different_fittings_center_of_mass, get_different_fittings_histogram, plot_cells, plot_nuclei_segmentations, fit_cells, get_distribution_histograms
from distribution_indexes import generate_distribution_indexes
import pandas as pd
from scipy import ndimage
import cv2

def read_multipage_tiff(file_path):
    img = Image.open(file_path)
    images = []
    while True:
        try:
            img.seek(len(images))  # Go to the next frame
            images.append(img.copy())
        except EOFError:
            break
    return images

def split_list_images(images, n_channels):
    list_list_images = []
    n_images = len(images)
    n_images_per_channel = int(n_images / n_channels)
    
    for i_channel in range(n_channels):
        images_channel = []
        #print('i_channel: ' + str(i_channel))
        for i_img_in_channel in range(n_images_per_channel):
            #print('i_img_in_channel: ' + str(i_img_in_channel))
            idx = i_img_in_channel * n_channels + i_channel
            #print(idx)
            img = images[idx]
            images_channel.append(img)
        list_list_images.append(images_channel)
        
    return list_list_images

def projected_image(images):
    # Convert images to numpy arrays
    images_array = np.array([np.array(img, dtype=np.float32) for img in images])

    # Compute the average of the images
    avg_image = np.mean(images_array, axis=0)

    # Convert back to uint8
    avg_image = np.uint16(avg_image)

    return avg_image

def function_debug(tiff_path):
    
    # Example usage
    # file_path = "multipage.tiff"
    images = read_multipage_tiff(tiff_path)
    list_list_images = split_list_images(images, n_channels)
    
    avg_img = projected_image(list_list_images[0])
    
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, 
                           constrained_layout = True)
    
    plt.subplot(1, 1, 1)
    plt.imshow(avg_img,cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
    
    
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, 
                           constrained_layout = True)
    
    
    plt.subplot(1, 1, 1)
    plt.imshow(images[0],cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
    
    
    plt.subplot(1, 1, 1)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, 
                           constrained_layout = True)
    
    plt.imshow(images[16],cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
    
    plt.subplot(1, 1, 1)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, 
                           constrained_layout = True)
    
    plt.imshow(np.abs(avg_img - images[16]),cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
    

#Main code

def create_folder_for_sample(path, sample_name):
    # Get the filename from the path
    folder_name = os.path.join(path, sample_name)
    
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def get_sample_name(path):
    head_tail = os.path.split(path)
    filename = head_tail[1]
    #print(filename)
    
    if filename.endswith('.tiff'):
        sample_name = filename[:-5]
    else: #end in .tif
        sample_name = filename[:-4]
    return sample_name


def main():
    
    plt.close('all')
    sample_name = get_sample_name(tiff_path)
    #print(sample_name)
    directory_path = os.path.dirname(tiff_path)
    #print(directory_path)
    folder_output = create_folder_for_sample(directory_path, sample_name)
    #print(folder_output)
    images = read_multipage_tiff(tiff_path)
    list_list_images = split_list_images(images, n_channels)
    
    numpydata_C1 = projected_image(list_list_images[0])
    numpydata_C2 = projected_image(list_list_images[1])
    numpydata_C3 = projected_image(list_list_images[2])
    numpydata_C4 = projected_image(list_list_images[3])
    
    #Load image (first channel)
    if flag_normalize:
        numpydata_C1 = functionPercNorm( np.single(numpydata_C1))
        numpydata_C2 = functionPercNorm( np.single(numpydata_C2))
        numpydata_C3 = functionPercNorm( np.single(numpydata_C3))
        numpydata_C4 = functionPercNorm( np.single(numpydata_C4))
    
    dims=numpydata_C1.shape
    
    #Load model
    model_trained_C1 = models.CellposeModel(pretrained_model=path_model_trained_C1, gpu=flag_gpu)
    print('Segment channel 1')
    #Segment channel 1
    numpydata_C1_segmentation, flows, styles = model_trained_C1.eval(numpydata_C1, diameter=None, channels= [[0,0]])
    
    cell_props_C1 = get_props_per_cell(numpydata_C1_segmentation)
    
    del model_trained_C1
    
    #Mask nuclei little dilated
    struct2 = ndimage.generate_binary_structure(2, 2)
    mask_nuclei = numpydata_C1_segmentation>0
    mask_nuclei = ndimage.binary_dilation(mask_nuclei, structure=struct2)
    
    #Load model
    model_trained_C2 = models.CellposeModel(pretrained_model=path_model_trained_C2, gpu=flag_gpu)
    print('Segment channel 2')
    numpydata_C2_segmentation, flows, styles = model_trained_C2.eval(numpydata_C2, diameter=None, channels= [[0,0]])
    
    cell_props_C2 = get_props_per_cell(numpydata_C2_segmentation)
    
    numpydata_C2_segmentation_match_nuclei = np.where(mask_nuclei,numpydata_C2_segmentation,0)
    #Get centroids and shape descriptors
    cell_props_C2_match_nuclei = get_props_per_cell(numpydata_C2_segmentation_match_nuclei)
    
    del model_trained_C2
    
    #Load model
    model_trained_C3 = models.CellposeModel(pretrained_model=path_model_trained_C3, gpu=flag_gpu)
    print('Segment channel 3')
    numpydata_C3_segmentation, flows, styles = model_trained_C3.eval(numpydata_C3, diameter=None, channels= [[0,0]])
    
    cell_props_C3 = get_props_per_cell(numpydata_C3_segmentation)
    
    numpydata_C3_segmentation_match_nuclei = np.where(mask_nuclei,numpydata_C3_segmentation,0)
    #Get centroids and shape descriptors
    cell_props_C3_match_nuclei = get_props_per_cell(numpydata_C3_segmentation_match_nuclei)
    
    del model_trained_C3
    
    #Load model
    model_trained_C4 = models.CellposeModel(pretrained_model=path_model_trained_C4, gpu=flag_gpu)
    print('Segment channel 4')
    numpydata_C4_segmentation, flows, styles = model_trained_C4.eval(numpydata_C4, diameter=None, channels= [[0,0]])
    
    cell_props_C4 = get_props_per_cell(numpydata_C4_segmentation)
    
    numpydata_C4_segmentation_match_nuclei = np.where(mask_nuclei,numpydata_C4_segmentation,0)
    #Get centroids and shape descriptors
    cell_props_C4_match_nuclei = get_props_per_cell(numpydata_C4_segmentation_match_nuclei)
    
    del model_trained_C4
    
    # Save the plot to a PNG file
    
    path_to_save = os.path.join(folder_output, sample_name + '_nuclei_segmentation.png')
    
    #plt.savefig(os.path.join(folder_output, sample_name + '_nuclei_segmentation.png'), dpi=300)
    fig, ax = plot_nuclei_segmentations(numpydata_C1, numpydata_C2, numpydata_C3, numpydata_C4,\
                                  numpydata_C1_segmentation, numpydata_C2_segmentation, numpydata_C3_segmentation, numpydata_C4_segmentation,\
                                      numpydata_C2_segmentation_match_nuclei,\
                                          numpydata_C3_segmentation_match_nuclei, numpydata_C4_segmentation_match_nuclei,\
                                              path_to_save, mask_nuclei = mask_nuclei)
    
    path_to_save = os.path.join(folder_output, sample_name + '_histograms.png')
    count_C1, count_C2, count_C3, count_C4 = get_distribution_histograms(cell_props_C1, cell_props_C2, cell_props_C3, cell_props_C4, dims, n_bins=20, path_to_save = path_to_save)
    
    #Generate output in CSV
    
    # dictionary of lists 
    dict = {'C1': count_C1, 'C2': count_C2, 'C3': count_C3, 'C4': count_C4} 
        
    df = pd.DataFrame(dict)
        
    #print(df)
    
    #Save to csv
    csv_output = os.path.join(folder_output, sample_name + '_bins.csv')
    df.to_csv(csv_output, index=False, sep = ',') 
    
    # #Save nuclei segmentation
    C1_segmentation_output = os.path.join(folder_output, sample_name + '_C1_segmentation.png')
    
    cv2.imwrite(C1_segmentation_output, numpydata_C1_segmentation)
    
    
    C2_segmentation_output = os.path.join(folder_output, sample_name + '_C2_segmentation.png')
    cv2.imwrite(C2_segmentation_output, numpydata_C2_segmentation)
    C2_segmentation_output = os.path.join(folder_output, sample_name + '_C2_segmentation_masked.png')
    cv2.imwrite(C2_segmentation_output, numpydata_C2_segmentation_match_nuclei)
    
    C3_segmentation_output = os.path.join(folder_output, sample_name + '_C3_segmentation.png')
    cv2.imwrite(C3_segmentation_output, numpydata_C3_segmentation)
    C3_segmentation_output = os.path.join(folder_output, sample_name + '_C3_segmentation_masked.png')
    cv2.imwrite(C3_segmentation_output, numpydata_C3_segmentation_match_nuclei)
    
    C4_segmentation_output = os.path.join(folder_output, sample_name + '_C4_segmentation.png')
    cv2.imwrite(C4_segmentation_output, numpydata_C4_segmentation)
    C4_segmentation_output = os.path.join(folder_output, sample_name + '_C4_segmentation_masked.png')
    cv2.imwrite(C4_segmentation_output, numpydata_C4_segmentation_match_nuclei)
    
    # Analyzing neural layer edge on the top
    if flag_histbins_for_outliers:
        C1_layer_nuclei = get_layer_nuclei_histogram(numpydata_C1_segmentation, count_C1)
        C2_layer_nuclei = get_layer_nuclei_histogram(numpydata_C2_segmentation_match_nuclei, count_C2)
        C3_layer_nuclei = get_layer_nuclei_histogram(numpydata_C3_segmentation_match_nuclei, count_C3)
        C4_layer_nuclei = get_layer_nuclei_histogram(numpydata_C4_segmentation_match_nuclei, count_C4)
    else:
        C1_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C1_segmentation, subimage_width = subimage_width, flag_show = False)
        C2_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C2_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C3_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C3_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
        C4_layer_nuclei = get_layer_nuclei_center_of_mass(numpydata_C4_segmentation_match_nuclei, subimage_width = subimage_width, flag_show = False)
    
    C3_top_cell_labels, C3_top_cells_xy = get_top_cells_labels(C3_layer_nuclei, subimage_width = subimage_width)
    C4_top_cell_labels, C4_top_cells_xy = get_top_cells_labels(C4_layer_nuclei, subimage_width = subimage_width)
    C2_top_cell_labels, C2_top_cells_xy = get_top_cells_labels(C2_layer_nuclei, subimage_width = subimage_width)
    
    C3_bottom_cell_labels, C3_bottom_cells_xy = get_top_cells_labels(C3_layer_nuclei, subimage_width = subimage_width, bottom_cells = True)
    C4_bottom_cell_labels, C4_bottom_cells_xy = get_top_cells_labels(C4_layer_nuclei, subimage_width = subimage_width, bottom_cells = True)
    C2_bottom_cell_labels, C2_bottom_cells_xy = get_top_cells_labels(C2_layer_nuclei, subimage_width = subimage_width, bottom_cells = True)
    
    path_to_save = os.path.join(folder_output, sample_name + '_edge_fitting.png')
    fig, ax = plot_nuclei_segmentations(numpydata_C1, numpydata_C2, numpydata_C3, numpydata_C4,\
                                  numpydata_C1_segmentation, numpydata_C2_segmentation, numpydata_C3_segmentation, numpydata_C4_segmentation,\
                                      numpydata_C2_segmentation_match_nuclei>0,\
                                          numpydata_C3_segmentation_match_nuclei>0, numpydata_C4_segmentation_match_nuclei>0,\
                                              path_to_save = None, mask_nuclei = mask_nuclei)
    
    
    
    plot_cells(ax[0,5], C2_top_cells_xy, marker_size = marker_size)
    plot_cells(ax[1,2], C3_top_cells_xy, marker_size = marker_size)
    plot_cells(ax[1,5], C4_top_cells_xy, marker_size = marker_size)
    
    
    plot_cells(ax[0,5], C2_bottom_cells_xy, color='green', marker_size = marker_size)
    plot_cells(ax[1,2], C3_bottom_cells_xy, color='green', marker_size = marker_size)
    plot_cells(ax[1,5], C4_bottom_cells_xy, color='green', marker_size = marker_size)
    
    
    C2_top_slope, C2_top_intercept, C2_top_r_value, C2_top_p_value, C2_top_std_err = fit_cells(C2_top_cells_xy)
    C3_top_slope, C3_top_intercept, C3_top_r_value, C3_top_p_value, C3_top_std_err = fit_cells(C3_top_cells_xy)
    C4_top_slope, C4_top_intercept, C4_top_r_value, C4_top_p_value, C4_top_std_err = fit_cells(C4_top_cells_xy)
    
    C2_bottom_slope, C2_bottom_intercept, C2_bottom_r_value, C2_bottom_p_value, C2_bottom_std_err = fit_cells(C2_bottom_cells_xy)
    C3_bottom_slope, C3_bottom_intercept, C3_bottom_r_value, C3_bottom_p_value, C3_bottom_std_err = fit_cells(C3_bottom_cells_xy)
    C4_bottom_slope, C4_bottom_intercept, C4_bottom_r_value, C4_bottom_p_value, C4_bottom_std_err = fit_cells(C4_bottom_cells_xy)
    
    
    data_fitting_bottom = {
            'r_value': [C2_bottom_r_value, C3_bottom_r_value, C4_bottom_r_value],
            'p_value': [C2_bottom_p_value, C3_bottom_p_value, C4_bottom_p_value],
            'std_error': [C2_bottom_std_err, C3_bottom_std_err, C4_bottom_std_err],
            'slope': [C2_bottom_slope, C3_bottom_slope, C4_bottom_slope],
            'intercept': [C2_bottom_intercept, C3_bottom_intercept, C4_bottom_intercept]
    }
    
    # Create DataFrame
    df_bottom = pd.DataFrame(data_fitting_bottom, index=['C2', 'C3', 'C4'])
    
    # Save DataFrame to CSV
    csv_output = os.path.join(folder_output, sample_name + '_bottom_edge_fitting.csv')
    df_bottom.to_csv(csv_output, index=False, sep = ',') 
    
    data_fitting_top = {
            'r_value': [C2_top_r_value, C3_top_r_value, C4_top_r_value],
            'p_value': [C2_top_p_value, C3_top_p_value, C4_top_p_value],
            'std_error': [C2_top_std_err, C3_top_std_err, C4_top_std_err],
            'slope': [C2_top_slope, C3_top_slope, C4_top_slope],
            'intercept': [C2_top_intercept, C3_top_intercept, C4_top_intercept]
    }
    
    # Create DataFrame
    df_top = pd.DataFrame(data_fitting_top, index=['C2', 'C3', 'C4'])
    
    # Save DataFrame to CSV
    csv_output = os.path.join(folder_output, sample_name + '_top_edge_fitting.csv')
    df_top.to_csv(csv_output, index=False, sep = ',') 
    
    # Different errors for different widths
    
    if flag_histbins_for_outliers:
        subimage_widths, C2_list_std_err, C3_list_std_err, C4_list_std_err, C2_list_r_value, C3_list_r_value, C4_list_r_value =\
            get_different_fittings_histogram(numpydata_C1_segmentation, numpydata_C2_segmentation_match_nuclei, numpydata_C3_segmentation_match_nuclei, numpydata_C4_segmentation_match_nuclei,\
                                             count_C1, count_C2, count_C3, count_C4)
    else:
        subimage_widths, C2_list_std_err, C3_list_std_err, C4_list_std_err, C2_list_r_value, C3_list_r_value, C4_list_r_value =\
            get_different_fittings_center_of_mass(numpydata_C1_segmentation, numpydata_C2_segmentation_match_nuclei, numpydata_C3_segmentation_match_nuclei, numpydata_C4_segmentation_match_nuclei)
    
    # Std-error
    fitting_top_cells_table = {'width': ['C2', 'C3', 'C4']}
    
    for i_width in range(len(subimage_widths)):
        key = str(subimage_widths[i_width])
        
        values = [C2_list_std_err[i_width], C3_list_std_err[i_width], C4_list_std_err[i_width]]
        
        fitting_top_cells_table[key] = values
    
    # Create DataFrame
    #df_distribution_indexes = pd.DataFrame(distribution_indexes_table, index=['C2','C3','C4'])
    df_fitting_top_cells = pd.DataFrame(fitting_top_cells_table)
    # Save DataFrame to CSV
    csv_output = os.path.join(folder_output, sample_name + '_top_edge_std_errors.csv')
    df_fitting_top_cells.to_csv(csv_output, index=False, sep = ',') 
    
    # R-value
    fitting_top_cells_table = {'width': ['C2', 'C3', 'C4']}

    for i_width in range(len(subimage_widths)):
        key = str(subimage_widths[i_width])
        
        values = [C2_list_r_value[i_width], C3_list_r_value[i_width], C4_list_r_value[i_width]]
        
        fitting_top_cells_table[key] = values
    
    # Create DataFrame
    #df_distribution_indexes = pd.DataFrame(distribution_indexes_table, index=['C2','C3','C4'])
    df_fitting_top_cells = pd.DataFrame(fitting_top_cells_table)
    # Save DataFrame to CSV
    csv_output = os.path.join(folder_output, sample_name + '_top_edge_r_value.csv')
    df_fitting_top_cells.to_csv(csv_output, index=False, sep = ',')
    
    
    #Draw top fitted line
    x = [0, dims[1]-2]
    C2_top_y = C2_top_slope*np.array(x) + C2_top_intercept
    ax[0,5].plot(x, C2_top_y, color=fitting_color, label='Fitted line',linewidth=marker_size/10)
    C3_top_y = C3_top_slope*np.array(x) + C3_top_intercept
    ax[1,2].plot(x, C3_top_y, color=fitting_color, label='Fitted line',linewidth=marker_size/10)
    C4_top_y = C4_top_slope*np.array(x) + C4_top_intercept
    ax[1,5].plot(x, C4_top_y, color=fitting_color, label='Fitted line',linewidth=marker_size/10)
    
    #Draw bottom fitted line
    #x = [0, dims[1]-2]
    C2_bottom_y = C2_bottom_slope*np.array(x) + C2_bottom_intercept
    ax[0,5].plot(x, C2_bottom_y, color=fitting_color, label='Fitted line',linewidth=marker_size/10)
    C3_bottom_y = C3_bottom_slope*np.array(x) + C3_bottom_intercept
    ax[1,2].plot(x, C3_bottom_y, color=fitting_color, label='Fitted line',linewidth=marker_size/10)
    C4_bottom_y = C4_bottom_slope*np.array(x) + C4_bottom_intercept
    ax[1,5].plot(x, C4_bottom_y, color=fitting_color, label='Fitted line',linewidth=marker_size/10)
    
    if flag_text:
        ax[0,5].text(x[0] + 10, C2_top_y[0] - text_shift, f'r-value: {C2_top_r_value:.3f} std-err: {C2_top_std_err:.3f}', color=fitting_color, fontsize=fontsize);
        ax[1,2].text(x[0] + 10, C3_top_y[0] - text_shift, f'r-value: {C3_top_r_value:.3f} std-err: {C3_top_std_err:.3f}', color=fitting_color, fontsize=fontsize);
        ax[1,5].text(x[0] + 10, C4_top_y[0] - text_shift, f'r-value: {C4_top_r_value:.3f} std-err: {C4_top_std_err:.3f}', color=fitting_color, fontsize=fontsize);
        
        ax[0,5].text(x[0] + 10, C2_bottom_y[0] - text_shift, f'r-value: {C2_bottom_r_value:.3f} std-err: {C2_bottom_std_err:.3f}', color=fitting_color, fontsize=fontsize);
        ax[1,2].text(x[0] + 10, C3_bottom_y[0] - text_shift, f'r-value: {C2_bottom_r_value:.3f} std-err: {C3_bottom_std_err:.3f}', color=fitting_color, fontsize=fontsize);
        ax[1,5].text(x[0] + 10, C4_bottom_y[0] - text_shift, f'r-value: {C2_bottom_r_value:.3f} std-err: {C4_bottom_std_err:.3f}', color=fitting_color, fontsize=fontsize);
    
    plt.savefig(path_to_save, dpi=400)
    
    #Build boxes where to compute the distribution indexes
    x_middle = int(dims[1]/2)
    
    C2_top_bb_y = C2_top_slope*np.array(x_middle) + C2_top_intercept
    C3_top_bb_y = C3_top_slope*np.array(x_middle) + C3_top_intercept
    C4_top_bb_y = C4_top_slope*np.array(x_middle) + C4_top_intercept
    
    C2_bottom_bb_y = C2_bottom_slope*np.array(x_middle) + C2_bottom_intercept
    C3_bottom_bb_y = C3_bottom_slope*np.array(x_middle) + C3_bottom_intercept
    C4_bottom_bb_y = C4_bottom_slope*np.array(x_middle) + C4_bottom_intercept
    
    # xmin, ymin, xmax, ymax = bbox
    C2_bbox = [0, C2_top_bb_y, dims[1], C2_bottom_bb_y]
    C3_bbox = [0, C3_top_bb_y, dims[1], C3_bottom_bb_y]
    C4_bbox = [0, C4_top_bb_y, dims[1], C4_bottom_bb_y]
    
    # Compute distribution indexes
    df_distribution_indexes, morisita_handle = generate_distribution_indexes(dims, cell_props_C1, cell_props_C2, cell_props_C3, cell_props_C4, C2_bbox, C3_bbox, C4_bbox,\
                                                                             dx_morisita = d_morisita, dy_morisita = d_morisita)
    
    # Save DataFrame to CSV
    csv_output = os.path.join(folder_output, sample_name + '_distribution_indexes.csv')
    df_distribution_indexes.to_csv(csv_output, index=None, sep = ',') 
    
    path_to_save = os.path.join(folder_output, sample_name + '_morisita_succession.png')
    morisita_handle.savefig(path_to_save, dpi=400)
    


if __name__ == "__main__":
    main()
