#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################   PARAMETERS   #########################
tiff_path = '' #In Windows, place an r before the ''

#Channel 1 is DAPI
path_model_channel1_nuclei_trained  = ''  #In Windows, place an r before the ''
path_model_channel2_trained  = ''  #In Windows, place an r before the ''
path_model_channel3_trained  = ''  #In Windows, place an r before the ''

#Can be changed
label_x = 'Channel 2'
label_y = 'Channel 3'
flag_normalize = False
flag_gpu = False
channels = [[0,0]] #Same channels as training
min_pixels_matching = 200
n_channels = 3

#Change if not desired naming
segmentation_dapi_nuclei_output        = tiff_path + '_nuclei_seg.png'
segmentation_channel2_output        = tiff_path + '_'+ label_x +'_seg.png'
segmentation_channel3_output        = tiff_path + '_'+ label_y +'_seg.png'


##############################################################################

import sys
import os
import cv2
import numpy as np
from cellpose import models
import matplotlib.pyplot as plt
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from quantify_segmentation import get_props_per_cell, get_joint_expr_per_cell, plot_expressions, save_csv
from aux_functions.functionPercNorm import functionPercNorm
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list, split_list_images, get_projected_image
from skimage import measure
import csv
#Move those readings to aux functions!!!

def segment_channel(img_original, path_model_trained, flag_normalize = False, gpu=False, segmentation_output_path = None, th_size=100):

    if flag_normalize:
        img_original = functionPercNorm( np.single(img_original))
    
    model_trained_channel = models.CellposeModel(pretrained_model=path_model_trained, gpu=flag_gpu)
    
    #Segment cells in image
    img_segmentation_channel, flows, styles = model_trained_channel.eval(img_original, diameter=None, channels= [[0,0]])
    
    #Get centroids and shape descriptors
    cell_props_channel = get_props_per_cell(img_segmentation_channel)
    
    if segmentation_output_path is not None:
        #Save membrane segmentation
        cv2.imwrite(segmentation_output_path, img_segmentation_channel)
        
    return img_segmentation_channel, cell_props_channel

def normalize_background(img_original, img_segmentation, flag_norm_type = 'Histogram'):
    background_values = img_original[img_segmentation == 0]
    mean_background = np.mean(background_values)
    
    #Background =1
    if flag_norm_type == 'Background': 
        
        normalized = np.array(img_original, dtype=float) / mean_background
    else:
        
        normalized = functionPercNorm( np.single(img_original))
    return normalized



# Not being used but keeping the code
def plot_expressions_coloured_by_prediction(list_cell_expr1_expr2, title_plot):
    max_expr = 0;
    plt.figure(figsize=(8, 8))
    
    handle_exp_not = None
    handle_exp1 = None
    handle_exp2 = None
    for cell_expr1_expr2 in list_cell_expr1_expr2:
        label, mean_exp1, mean_exp2, pred1, pred2 = cell_expr1_expr2
        if max_expr< np.max([mean_exp1,mean_exp2]):
            max_expr = np.max([mean_exp1,mean_exp2])
            
        if pred1 == 1:
            handle_exp1 = plt.scatter(mean_exp1, mean_exp2, color='red', marker='o')
        else:
            if pred2 == 1:
                handle_exp2 = plt.scatter(mean_exp1, mean_exp2, color='green', marker='o')
            else:
                handle_exp_not = plt.scatter(mean_exp1, mean_exp2, color='black', marker='o')
            
    plt.title(title_plot)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.ylim(0, max_expr + 0.01)
    plt.xlim(0, max_expr + 0.01)
    if handle_exp_not is not None:
        plt.legend(handles=[handle_exp1, handle_exp2, handle_exp_not], labels=['Pred as 2','Pred as 3', 'Not Pred'])
    else:
        plt.legend(handles=[handle_exp1, handle_exp2], labels=['Pred as 2','Pred as 3'])
    #plt.show()

def show_images(img_channel1, img_channel2, title_plot):
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")

    plt.subplot(1, 2, 1)
    plt.imshow(img_channel1, cmap='gray')
    plt.axis('off')
    plt.title(title_plot)

    plt.subplot(1, 2, 2)
    plt.imshow(img_channel2, cmap='gray')
    plt.axis('off')
    plt.title(title_plot)

    #plt.show()


    
def main():
    images = read_multipage_tiff_as_list(tiff_path)
    list_list_images = split_list_images(images, n_channels)
    
    #Read tiff stack of 3 channels
    
    #Split
    #Numbering of variables starting in 1 for reader/user of the code
    img_channel1_original = get_projected_image(list_list_images[0])
    img_channel2_original = get_projected_image(list_list_images[1])
    img_channel3_original = get_projected_image(list_list_images[2])
    
    img_segmentation_nuclei, cell_props_nuclei      = segment_channel(img_channel1_original, path_model_channel1_nuclei_trained,    flag_normalize = flag_normalize, gpu=flag_gpu, segmentation_output_path = None)
    img_segmentation_channel2, cell_props_channel2  = segment_channel(img_channel2_original, path_model_channel2_trained,           flag_normalize = flag_normalize, gpu=flag_gpu, segmentation_output_path = None)
    img_segmentation_channel3, cell_props_channel3  = segment_channel(img_channel3_original, path_model_channel3_trained,           flag_normalize = flag_normalize, gpu=flag_gpu, segmentation_output_path = None)
    
    #cell_props is the one to save if they want geometric descriptors
    
    plt.close('all')
    
    
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
    
    #plt.show()
    plt.savefig(tiff_path + '_segmentations.png', dpi=400)
    
    
    list_cell_expr1_expr2 = get_joint_expr_per_cell(img_segmentation_nuclei, img_channel2_original, img_channel3_original, img_segmentation_channel2, img_segmentation_channel3)
    save_csv(list_cell_expr1_expr2, tiff_path + '_orig_values.csv')
    
    plot_expressions(list_cell_expr1_expr2, title_plot = 'Original values')
    plt.savefig(tiff_path + '_plot_orig_values.png', dpi=400)
    
    img_channel2_norm = normalize_background(img_channel2_original, img_segmentation_nuclei)
    img_channel3_norm = normalize_background(img_channel3_original, img_segmentation_nuclei)
    
    show_images(img_channel2_norm, img_channel3_norm, title_plot = 'Histogram normalization')
    plt.savefig(tiff_path + '_imgs_hist_norm.png', dpi=400)
    
    list_cell_expr1_expr2 = get_joint_expr_per_cell(img_segmentation_nuclei,\
                                                    img_channel2_norm, img_channel3_norm,\
                                                    img_segmentation_channel2, img_segmentation_channel3)
    save_csv(list_cell_expr1_expr2, tiff_path + '_hist_norm_values.csv')
    
    plot_expressions(list_cell_expr1_expr2, title_plot = 'Histogram normalization')
    plt.savefig(tiff_path + '_plot_hist_norm.png', dpi=400)
    
    img_channel2_norm = normalize_background(img_channel2_original, img_segmentation_nuclei, flag_norm_type = 'Background')
    img_channel3_norm = normalize_background(img_channel3_original, img_segmentation_nuclei, flag_norm_type = 'Background')
    
    show_images(img_channel2_norm, img_channel3_norm, title_plot = 'Background normalization')
    plt.savefig(tiff_path + '_imgs_back_norm.png', dpi=400)
    
    list_cell_expr1_expr2 = get_joint_expr_per_cell(img_segmentation_nuclei,\
                                                    img_channel2_norm, img_channel3_norm,\
                                                    img_segmentation_channel2, img_segmentation_channel3)
    save_csv(list_cell_expr1_expr2, tiff_path + '_back_norm_values.csv')
    
    plot_expressions(list_cell_expr1_expr2, title_plot = 'Background normalization')
    plt.savefig(tiff_path + '_plot_back_norm.png', dpi=400)

    
if __name__ == "__main__":
    main()
