#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################   PARAMETERS   #########################

folder_input = '' # Write the folder path, without the last \ or /

path_model_trained = 'pericytes_bodies_c1pericytes_c2vessel_model_nuclei_diam15_ji0.1488.112184' # Complete with the full path of the file, downloaded from OneDrive


# File endings can be changed
folder_output = folder_input + '_segmentations_20250212'
ending_segmentation_binary = '_segmentation_binary.tiff'
ending_segmentation = '_segmentation.tiff'
ending_csv_properties = '_pericytes_properties.csv'
ending_volume_pericytes = '_volume_pericytes.tiff'
ending_volume_vessel = '_volume_vessel.tiff'

flag_gpu = False # True if GPU is correctly installed

##############################################################################

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from cellpose import models
import numpy as np
import time
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list, split_list_images
from aux_functions.functionSaveTIFFMultipage import functionSaveTIFFMultipage
from aux_functions.functions_volume_analysis import get_objects_properties, plot_save_sphericity_histogram, plot_save_length_histogram

# Model channels: [3, 2] Where 3 was pericytes (our nuclei) and 2 the vessel (our cyto)
diameter = None
channels = [1,2]

def main():
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    
    model_trained = models.CellposeModel(pretrained_model=path_model_trained, gpu=flag_gpu)
    
    file_images = []
    for file in os.listdir(folder_input):
        if file.endswith(".tiff") or file.endswith(".tif"):
            file_images.append(file)
    
    for file_image in file_images:
        
        print('---- Segmenting: ' + file_image + ' ----')
        start_time = time.time()
        
        path_image = os.path.join(folder_input, file_image)
        
        images = read_multipage_tiff_as_list(path_image)
        list_list_images = split_list_images(images, 2)
        channel_pericytes = list_list_images[1]
        channel_vessel = list_list_images[0]
        
        channel_pericytes_np = [np.asarray(img) for img in channel_pericytes]
        channel_vessel_np = [np.asarray(img) for img in channel_vessel]
        
        channel_pericytes_np = np.stack(channel_pericytes_np, axis=-1)
        channel_vessel_np = np.stack(channel_vessel_np, axis=-1)
        
        # Z x nchan x Y x X.
        functionSaveTIFFMultipage(channel_pericytes_np, os.path.join(folder_output,file_image + ending_volume_pericytes), 8)
        functionSaveTIFFMultipage(channel_vessel_np, os.path.join(folder_output,file_image + ending_volume_vessel), 8)
        
        # Stack along a new axis to create a (X, Y, Z, nchannels) array
        vol_4d = np.stack([channel_pericytes_np, channel_vessel_np], axis=-1)  # Shape: (X, Y, Z, 2)
        
        # Transpose to get the desired shape: (Z, nchannels, Y, X)
        vol_4d = vol_4d.transpose(2, 3, 1, 0)  # Shape: (Z, 2, Y, X)
        
        masks, flows, styles = model_trained.eval(vol_4d, channels=channels, diameter=diameter, do_3D = True)
        
        masks = np.uint16(masks)
        n_bodies = len(np.unique(masks)) - 1 #0 is the background
        print('Number of cell bodies: ' + str(n_bodies))
        df = get_objects_properties(masks)
        plot_save_sphericity_histogram(df, os.path.join(folder_output,file_image + '_sphericity_histogram.png'))
        plot_save_length_histogram(df, os.path.join(folder_output,file_image + '_length_histogram.png'))
        
        df.to_csv(os.path.join(folder_output,file_image + ending_csv_properties))
        
        functionSaveTIFFMultipage(np.transpose(masks.copy(), (2, 1, 0)), os.path.join(folder_output,file_image + ending_segmentation), 8)
        
        masks = np.uint8(masks>0) * 255;
        functionSaveTIFFMultipage(np.transpose(masks.copy(), (2, 1, 0)), os.path.join(folder_output,file_image + ending_segmentation_binary), 8)
        
        print("It took: %s seconds" % round((time.time() - start_time),1))
        
        print('--------')
    
if __name__ == "__main__":
    main()
