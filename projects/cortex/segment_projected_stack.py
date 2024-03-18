#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
###################################   PARAMETERS   #########################
############################################################################

#File path to .tif or .tiff file with a stitched stack of IHC images
tiff_path = '' #.tif file

# Number of channels in tiff stack
n_channels = 4

#Number of bins for histogram
n_bins = 20

#Path to trained architectures
path_model_trained_C1  = ''#'Neurons_C1.183326'
path_model_trained_C2  = ''#'Neurons_C2.919883'
path_model_trained_C3  = ''#'Neurons_C3.981474'
path_model_trained_C4  = ''#'Neurons_C4.909737'

#Parameters for running the segmentation
flag_normalize = False
flag_gpu = False

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
from quantify_segmentation import get_props_per_cell, get_density_bins
from aux_functions.functionPercNorm import functionPercNorm
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
    
    plt.rcParams["figure.autolayout"] = True
    
    fig, ax = plt.subplots(2, 6, figsize=(12, 6), sharex=True, sharey=True)
    
    plt.subplot(2, 6, 1)
    plt.imshow(numpydata_C1,cmap='gray')
    plt.gca().set_title('C1')
    
    plt.subplot(2, 6, 2)
    plt.imshow(mask_nuclei,cmap='gray')
    plt.gca().set_title('C1 - mask for Cs')
    
    plt.subplot(2, 6, 3)
    plt.imshow(numpydata_C1_segmentation,cmap='gist_ncar')
    plt.gca().set_title('C1 Seg')
    
    #Channel 2
    
    plt.subplot(2, 6, 4)
    plt.imshow(numpydata_C2,cmap='gray')
    plt.gca().set_title('C2')
    
    plt.subplot(2, 6, 5)
    plt.imshow(numpydata_C2_segmentation,cmap='gist_ncar')
    plt.gca().set_title('C2 Seg')
    
    plt.subplot(2, 6, 6)    
    plt.imshow(numpydata_C2_segmentation_match_nuclei,cmap='gist_ncar')
    plt.gca().set_title('C2 Seg - masked')
    
    #Channel 3
    
    plt.subplot(2, 6, 7)
    plt.imshow(numpydata_C3,cmap='gray')
    plt.gca().set_title('C3')
    
    plt.subplot(2, 6, 8)
    plt.imshow(numpydata_C3_segmentation,cmap='gist_ncar')
    plt.gca().set_title('C3 Seg')
    
    plt.subplot(2, 6, 9)
    plt.imshow(numpydata_C3_segmentation_match_nuclei,cmap='gist_ncar')
    plt.gca().set_title('C3 Seg - masked')
    
    #Channel 4
    
    plt.subplot(2, 6, 10)
    plt.imshow(numpydata_C4,cmap='gray')
    plt.gca().set_title('C4')
    
    plt.subplot(2, 6, 11)
    plt.imshow(numpydata_C4_segmentation,cmap='gist_ncar')
    plt.gca().set_title('C4 Seg')
    
    plt.subplot(2, 6, 12)
    plt.imshow(numpydata_C4_segmentation_match_nuclei,cmap='gist_ncar')
    plt.gca().set_title('C4 Seg - Masked')
    
    figManager = plt.get_current_fig_manager()
        
    figManager.window.showMaximized()
    
    #plt.figure(figsize=(8, 6))
    #plt.show()
    
    # Save the plot to a PNG file
    plt.savefig(os.path.join(folder_output, sample_name + '_nuclei_segmentation.png'), dpi=300)
    
    plt.close('all')
    
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
        
    figManager.window.showMaximized()
    #plt.figure(figsize=(8, 6))
    #plt.show()
    
    # Save the plot to a PNG file
    plt.savefig(os.path.join(folder_output, sample_name + '_histograms.png'), dpi=300)
    
    plt.close('all')
    
    #Generate output in CSV
    
    # dictionary of lists 
    dict = {'C1': count_C1, 'C2': count_C2, 'C3': count_C3, 'C4': count_C4} 
        
    df = pd.DataFrame(dict)
        
    #print(df)
    
    #Save to csv
    csv_output = os.path.join(folder_output, sample_name + '_bins.csv')
    #print('----------------------')
    #print(csv_output)
    df.to_csv(csv_output, index=False, sep = ',') 
    
    # #Save nuclei segmentation
    # cv2.imwrite(numpydata_C1_segmentation, )
    C1_segmentation_output = os.path.join(folder_output, sample_name + '_C1_segmentation.png')
    
    #print(C1_segmentation_output)
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

    
if __name__ == "__main__":
    main()