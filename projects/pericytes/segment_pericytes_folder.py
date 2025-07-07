#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################   PARAMETERS   #########################

folder_input = '' # Write the folder path, without the last \ or / . In Windows, place an r before the ''.

folder_models = '' # Folder where all your images must be. . In Windows, place an r before the ''.
model_name_pericytes_on_vessel =  'pericytes_touching_vessel_dilated5_model_nuclei_diam_15_ji_0.2169.511209' # name of the file, downloaded from OneDrive.
model_name_all_positive_pericytes = 'all_positives_pericytes_dilated41_model_cyto2_diam_12_ji_0.294.550827' # name of the file, downloaded from OneDrive.

# File endings can be changed
subfolder_output ='segmentations_20250707'
ending_segmentation_binary = '_segmentation_binary.tiff'
ending_positives_binary = '_positives_segmentation_binary.tiff'
ending_segmentation = '_segmentation.tiff'
ending_csv_properties = '_pericytes_properties.csv'
ending_volume_pericytes = '_volume_pericytes.tiff'
ending_volume_vessel = '_volume_vessel.tiff'

flag_gpu = False # True if GPU is correctly installed

channel_pericytes = 2 # Is the first or second channel in the czi file?

##############################################################################
##############################################################################
##############################################################################

diameter = None
#
channels_pericytes_in_vessel = [1,2]# [1,2]
dilation_pericytes_in_vessel = 5

channels_all_pericytes = [0, 0]
dilation_all_pericytes = 41

n_bins_debug = 10
th_volume = 400

##############################################################################

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # Check In Windows the \ or /
sys.path.append(root_path)
from cellpose import models
import numpy as np
import time
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list, split_list_images
from aux_functions.functionSaveTIFFMultipage import functionSaveTIFFMultipage
from aux_functions.functions_volume_analysis import get_objects_properties, plot_save_sphericity_histogram, plot_save_length_histogram
from pericytes_functions.functions_volume_analysis import get_objects_properties, plot_save_sphericity_histogram, plot_save_length_histogram, plot_save_volume_histogram
from pericytes_functions.template_paraview_python_script import TEMPLATE_HEAD
from scipy.ndimage import grey_dilation
import czifile
from quantify_segmentation import matching_label_pairs_perc

# Model channels: [3, 2] Where 3 was pericytes (our nuclei) and 2 the vessel (our cyto)
diameter = None
channels = [1,2]

def dilate_channel_in_volume(vol_4d, channel_to_dilate, kernel_size):
    vol_4d_dilated = vol_4d.copy()
    channel_1 = vol_4d_dilated[:, channel_to_dilate, :, :]

    # Define a 3D structuring element (e.g., 3x3x3 cube)
    struct = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
    
    # Apply 3D dilation
    dilated_channel = grey_dilation(channel_1, footprint=struct)
    
    # If needed, convert back to the same dtype as original
    dilated_channel = dilated_channel.astype(vol_4d_dilated.dtype)
    
    # Replace the original second channel with the dilated version
    vol_4d_dilated[:, channel_to_dilate, :, :] = dilated_channel
    
    return vol_4d_dilated

def main():
    
    path_model_trained_pericytes_on_vessel = os.path.join(folder_models,model_name_pericytes_on_vessel)
    path_model_trained_all_positive_pericytes = os.path.join(folder_models,model_name_all_positive_pericytes)
    
    if not (os.path.isfile(path_model_trained_pericytes_on_vessel) and os.path.isfile(path_model_trained_all_positive_pericytes)):
        print("Error: One or more trained models were not found.")
        sys.exit(1)
    
    #subfolder_output = 'segmentations'# + str(dilation_pericytes_in_vessel)
    folder_output = os.path.join(folder_input, subfolder_output)

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    file_images = []
    for file in os.listdir(folder_input):
        if file.endswith(".czi"):
            print(file)
            file_images.append(file)

    # file_images = [file_images[0]]
    for file_image in file_images:
        
        print('---- Segmenting: ' + file_image + ' ----')
        start_time = time.time()
        
        path_image = os.path.join(folder_input, file_image)
        
        folder_output_sample = os.path.join(folder_output, file_image + '_pericytesInVessel' + str(channels_pericytes_in_vessel) + '_pericytesAll' + str(channels_all_pericytes))
        
        if not os.path.exists(folder_output_sample):
            os.makedirs(folder_output_sample)
        
        czi = czifile.CziFile(path_image)
        # Extract metadata
        metadata = czi.metadata(raw=False)
        image_data = czi.asarray()
        print('Original czi: ' + str(image_data.shape))
        image_data = np.squeeze(image_data)
        print('Squeezed czi: ' + str(image_data.shape))
        shape = image_data.shape
        n_images = shape[0]
        
        if channel_pericytes == 2:
            channel_pericytes_np = image_data[1,:,:,:]
            channel_vessel_np =  image_data[0,:,:,:]
        else:
            channel_pericytes_np = image_data[0,:,:,:]
            channel_vessel_np =  image_data[1,:,:,:]
        del image_data, czi
        print('channel_pericytes_np: ' + str(channel_pericytes_np.shape))
        print('channel_vessel_np: ' + str(channel_vessel_np.shape))
        # Now it is the name of the output
        
        output_name = file_image + '_1_noFiltering_'
        
        # Z x nchan x Y x X.
        fullpath_tiff_pericytes_channel = os.path.join(folder_output_sample,output_name + ending_volume_pericytes)
        fullpath_tiff_vessel_channel = os.path.join(folder_output_sample,output_name + ending_volume_vessel)
        functionSaveTIFFMultipage(channel_pericytes_np, fullpath_tiff_pericytes_channel, 8)
        functionSaveTIFFMultipage(channel_vessel_np, fullpath_tiff_vessel_channel, 8)
        
        # Stack along a new axis to create a (X, Y, Z, nchannels) array
        vol_4d = np.stack([channel_pericytes_np, channel_vessel_np], axis=-1)  # Shape: (X, Y, Z, 2)
        # del channel_pericytes_np, channel_vessel_np #, channel_vessel_np
        # print('vol_4d.shape stacked 2 channels')
        # print(vol_4d.shape)

        vol_4d = vol_4d.transpose(0, 3, 1, 2)
        tuple_traspose_to_save = (0, 1, 2)
        
        # print('vol_4d.shape trasposed')
        # print(vol_4d.shape)
        vol_4d_dilated_vessel_channel = dilate_channel_in_volume(vol_4d, 1, kernel_size = dilation_pericytes_in_vessel)
        model_trained_pericytes_on_vessel = models.CellposeModel(pretrained_model=path_model_trained_pericytes_on_vessel, gpu=flag_gpu)
        masks_pericytes_on_vessel, _, _ = model_trained_pericytes_on_vessel.eval(vol_4d_dilated_vessel_channel, channels=channels_pericytes_in_vessel, diameter=diameter, do_3D = True)
        print('--Segmented pericytes on vessel--')
        del model_trained_pericytes_on_vessel
        
        vol_4d_dilated_vessel_channel = dilate_channel_in_volume(vol_4d, 1, kernel_size = dilation_all_pericytes)
        del vol_4d
        model_trained_all_positive_pericytes = models.CellposeModel(pretrained_model=path_model_trained_all_positive_pericytes, gpu=flag_gpu)
        masks_pericytes_all_positive_pericytes, _, _ = model_trained_all_positive_pericytes.eval(vol_4d_dilated_vessel_channel, channels=channels_all_pericytes, diameter=diameter, do_3D = True)
        print('--Segmented all pericytes on vessel--')
        del model_trained_all_positive_pericytes, vol_4d_dilated_vessel_channel
        
        ############# Analysis of pericytes segmentation #############
        print('--Analysis of pericytes segmentation--')
        masks_pericytes_on_vessel = np.uint16(masks_pericytes_on_vessel)
        n_bodies = len(np.unique(masks_pericytes_on_vessel)) - 1 #0 is the background
        print('Number of cell bodies: ' + str(n_bodies))
        df = get_objects_properties(masks_pericytes_on_vessel)
        plot_save_sphericity_histogram(df, os.path.join(folder_output_sample,output_name + '_sphericity_histogram_pericytes_on_vessel.png'), bins = n_bins_debug)
        plot_save_length_histogram(df, os.path.join(folder_output_sample,output_name + '_length_histogram_pericytes_on_vessel.png'), bins = n_bins_debug)
        plot_save_volume_histogram(df, os.path.join(folder_output_sample,output_name + '_volume_histogram_pericytes_on_vessel.png'), bins = n_bins_debug)
        df.to_csv(os.path.join(folder_output_sample,output_name + '_pericytes_on_vessel_' + ending_csv_properties))
        
        functionSaveTIFFMultipage(np.transpose(masks_pericytes_on_vessel.copy(), tuple_traspose_to_save), os.path.join(folder_output_sample,output_name + ending_segmentation), 8)
        
        del df
        masks_pericytes_all_positive_pericytes = np.uint16(masks_pericytes_all_positive_pericytes)
        n_positives = len(np.unique(masks_pericytes_all_positive_pericytes)) - 1 #0 is the background
        print('Number of positive pericytes: ' + str(n_positives))
        _,pericytes_positives_match,pericytes_in_both_volumes = matching_label_pairs_perc(masks_pericytes_all_positive_pericytes, masks_pericytes_on_vessel )
        
        
        df_all_positive = get_objects_properties(masks_pericytes_all_positive_pericytes)
        plot_save_sphericity_histogram(df_all_positive, os.path.join(folder_output_sample,output_name + '_sphericity_histogram_all_positive_pericytes.png'), bins = n_bins_debug)
        plot_save_length_histogram(df_all_positive, os.path.join(folder_output_sample,output_name + '_length_histogram_all_positive_pericytes.png'), bins = n_bins_debug)
        plot_save_volume_histogram(df_all_positive, os.path.join(folder_output_sample,output_name + '_volume_histogram_all_positive_pericytes.png'), bins = n_bins_debug)
        df_all_positive.to_csv(os.path.join(folder_output_sample,output_name + '_all_positive_pericytes_' + ending_csv_properties))
        
        functionSaveTIFFMultipage(np.transpose(masks_pericytes_on_vessel.copy(), tuple_traspose_to_save), os.path.join(folder_output_sample,output_name + ending_segmentation), 8)
        
        # Save binary volume (good for visualization)
        fullpath_tiff_pericytes_segmentation_binary = os.path.join(folder_output_sample,output_name + ending_segmentation_binary)
        
        functionSaveTIFFMultipage(np.transpose(np.uint8(masks_pericytes_on_vessel.copy()>0) * 255, tuple_traspose_to_save), fullpath_tiff_pericytes_segmentation_binary, 8)
        
        n_positives_not_matching = sum(1 for pair in pericytes_positives_match if pair[1] == 0)
        n_matching = len(pericytes_in_both_volumes)
        ratio_bodies_over_all_positives = n_bodies / (n_positives + 0.00001)
        print('Number of positive pericytes not matching: ' + str(n_positives_not_matching))
        print('Number of positive and pericytes matching: ' + str(n_matching))
        
        # Save binary volume (good for visualization)
        fullpath_tiff_pericytes_all_positives_binary = os.path.join(folder_output_sample,output_name + ending_positives_binary)
        functionSaveTIFFMultipage(np.transpose(np.uint8(masks_pericytes_all_positive_pericytes.copy()>0) * 255, tuple_traspose_to_save), fullpath_tiff_pericytes_all_positives_binary, 8)
        
        ############### Outputs in txt #############
        
        output_stats_path = os.path.join(folder_output_sample, output_name + "_stats.txt")
        with open(output_stats_path, "w") as f:
            f.write('Number of cell bodies (pericytes in vessel): ' + str(n_bodies))
            f.write('\nNumber of positive pericytes: ' + str(n_positives))
            f.write('\nNumber of positive pericytes not matching: ' + str(n_positives_not_matching))
            f.write('\nNumber of positive and pericytes matching: ' + str(n_matching))
            f.write('\nRatio bodies over all positives: ' + f"{ratio_bodies_over_all_positives:.3f}")
        
        ################ Paraview Script #############
        
        print("It took: %s seconds" % round((time.time() - start_time),1))
        print("Generating Paraview Python Script")
        
        colormap_vessel_channel = "Cool to Warm"
        colormap_pericytes_channel = "Viridis (matplotlib)"
        colormap_pericytes_segmentation = "X Ray"
        colormap_all_positives = "Black, Blue and White"
        volume_list = [
        (fullpath_tiff_vessel_channel, 'vessel_channel', colormap_vessel_channel),
        (fullpath_tiff_pericytes_channel, 'pericytes_channel', colormap_pericytes_channel),
        (fullpath_tiff_pericytes_segmentation_binary, 'pericytes_segmentation', colormap_pericytes_segmentation),
        (fullpath_tiff_pericytes_all_positives_binary, 'pericytes_all_positives', colormap_all_positives),
        ]
        
        output_script_path = os.path.join(folder_output_sample, output_name + "_render_volumes.py")
        
        load_calls = ""
        for tiff_path, registration_name, colormap in volume_list:
            tiff_escaped = tiff_path.replace("\\", "\\\\")  # escape backslashes for Windows
            load_calls += f"load_volume('{tiff_escaped}', '{registration_name}', '{colormap}')\n"
        
        # Combine all parts
        final_script = TEMPLATE_HEAD + load_calls
        
        # Write the generated script to a file
        with open(output_script_path, "w") as f:
            f.write(final_script)
        
        ################ FILTERING #############
        del n_bodies, n_positives, fullpath_tiff_pericytes_channel, fullpath_tiff_vessel_channel
        del fullpath_tiff_pericytes_segmentation_binary, fullpath_tiff_pericytes_all_positives_binary
        del n_positives_not_matching, n_matching, pericytes_in_both_volumes
        
        print('----Filtering all pericytes----')
        
        output_name = file_image + '_2_afterFiltering_'
        
        # Clear masks_pericytes_all_positive_pericytes of small masks_pericytes_all_positive_pericytes
        
        masks_pericytes_all_positive_pericytes_temp = np.zeros_like(masks_pericytes_all_positive_pericytes)
        unique_labels_all_positive = np.unique(masks_pericytes_all_positive_pericytes)
        unique_labels_all_positive = unique_labels_all_positive[1:] # Remove 0 (background)
        
        for label in unique_labels_all_positive:
            volume_with_one_pericyte = masks_pericytes_all_positive_pericytes == label
            num_positive_voxels = np.count_nonzero(volume_with_one_pericyte)
            if num_positive_voxels > th_volume:
                masks_pericytes_all_positive_pericytes_temp[volume_with_one_pericyte] = label
        
        del masks_pericytes_all_positive_pericytes # Delete unfiltered all pericytes
        masks_pericytes_all_positive_pericytes = masks_pericytes_all_positive_pericytes_temp
        del masks_pericytes_all_positive_pericytes_temp
        
        pericytes_all_positive_pericytes_in_vessel = np.zeros_like(masks_pericytes_all_positive_pericytes)
        
        _,_,pericytes_in_both_volumes = matching_label_pairs_perc(masks_pericytes_all_positive_pericytes, masks_pericytes_on_vessel )
        
        for allPericytes_inVessel in pericytes_in_both_volumes:
            # print(allPericytes_inVessel)
            volume_with_one_pericyte = masks_pericytes_all_positive_pericytes == allPericytes_inVessel[0]
            num_positive_voxels = np.count_nonzero(volume_with_one_pericyte)
            if num_positive_voxels > th_volume:
                pericytes_all_positive_pericytes_in_vessel[volume_with_one_pericyte] = allPericytes_inVessel[0]
        
        del masks_pericytes_on_vessel # Delete unfiltered pericytes in vessel
        
        ############# Analysis of filtered pericytes segmentation #############
        masks_pericytes_on_vessel = pericytes_all_positive_pericytes_in_vessel #Replace the mask_pericytes
        
        masks_pericytes_on_vessel = np.uint16(masks_pericytes_on_vessel)
        n_bodies = len(np.unique(masks_pericytes_on_vessel)) - 1 #0 is the background
        print('Number of cell bodies: ' + str(n_bodies))
        df = get_objects_properties(masks_pericytes_on_vessel)
        plot_save_sphericity_histogram(df, os.path.join(folder_output_sample,output_name + '_sphericity_histogram_pericytes_on_vessel.png'), bins = n_bins_debug)
        plot_save_length_histogram(df, os.path.join(folder_output_sample,output_name + '_length_histogram_pericytes_on_vessel.png'), bins = n_bins_debug)
        plot_save_volume_histogram(df, os.path.join(folder_output_sample,output_name + '_volume_histogram_pericytes_on_vessel.png'), bins = n_bins_debug)
        df.to_csv(os.path.join(folder_output_sample,output_name + ending_csv_properties))
        
        functionSaveTIFFMultipage(np.transpose(masks_pericytes_on_vessel.copy(), tuple_traspose_to_save), os.path.join(folder_output_sample,output_name + ending_segmentation), 8)
        
        masks_pericytes_all_positive_pericytes = np.uint16(masks_pericytes_all_positive_pericytes)
        n_positives = len(np.unique(masks_pericytes_all_positive_pericytes)) - 1 #0 is the background
        print('Number of positive pericytes: ' + str(n_positives))
        _,pericytes_positives_match,pericytes_in_both_volumes = matching_label_pairs_perc(masks_pericytes_all_positive_pericytes, masks_pericytes_on_vessel )
        
        # Save binary volume (good for visualization)
        fullpath_tiff_pericytes_segmentation_binary = os.path.join(folder_output_sample,output_name + ending_segmentation_binary)
        
        functionSaveTIFFMultipage(np.transpose(np.uint8(masks_pericytes_on_vessel.copy()>0) * 255, tuple_traspose_to_save), fullpath_tiff_pericytes_segmentation_binary, 8)
        
        n_positives_not_matching = sum(1 for pair in pericytes_positives_match if pair[1] == 0)
        n_matching = len(pericytes_in_both_volumes)
        ratio_bodies_over_all_positives = n_bodies / (n_positives + 0.00001)
        print('Number of positive pericytes not matching: ' + str(n_positives_not_matching))
        print('Number of positive and pericytes matching: ' + str(n_matching))
        
        # Save binary volume (good for visualization)
        fullpath_tiff_pericytes_all_positives_binary = os.path.join(folder_output_sample,output_name + ending_positives_binary)
        
        functionSaveTIFFMultipage(np.transpose(np.uint8(masks_pericytes_all_positive_pericytes.copy()>0) * 255, tuple_traspose_to_save), fullpath_tiff_pericytes_all_positives_binary, 8)
        
        fullpath_tiff_pericytes_channel = os.path.join(folder_output_sample,output_name + ending_volume_pericytes)
        fullpath_tiff_vessel_channel = os.path.join(folder_output_sample,output_name + ending_volume_vessel)
        functionSaveTIFFMultipage(channel_pericytes_np, fullpath_tiff_pericytes_channel, 8)
        functionSaveTIFFMultipage(channel_vessel_np, fullpath_tiff_vessel_channel, 8)
        
        
        volume_list = [
        (fullpath_tiff_vessel_channel, 'vessel_channel', colormap_vessel_channel),
        (fullpath_tiff_pericytes_channel, 'pericytes_channel', colormap_pericytes_channel),
        (fullpath_tiff_pericytes_segmentation_binary, 'pericytes_segmentation', colormap_pericytes_segmentation),
        (fullpath_tiff_pericytes_all_positives_binary, 'pericytes_all_positives', colormap_all_positives),
        ]
        
        output_script_path = os.path.join(folder_output_sample, output_name + "_render_volumes.py")
        
        load_calls = ""
        for tiff_path, registration_name, colormap in volume_list:
            tiff_escaped = tiff_path.replace("\\", "\\\\")  # escape backslashes for Windows
            load_calls += f"load_volume('{tiff_escaped}', '{registration_name}', '{colormap}')\n"
        
        # Combine all parts
        final_script = TEMPLATE_HEAD + load_calls
            
        # Write the generated script to a file
        with open(output_script_path, "w") as f:
            f.write(final_script)
        
        ############### Outputs in txt #############
        
        output_stats_path = os.path.join(folder_output_sample, output_name + "_stats.txt")
        with open(output_stats_path, "w") as f:
            f.write('Number of cell bodies (pericytes in vessel): ' + str(n_bodies))
            f.write('\nNumber of positive pericytes: ' + str(n_positives))
            f.write('\nNumber of positive pericytes not matching: ' + str(n_positives_not_matching))
            f.write('\nNumber of positive and pericytes matching: ' + str(n_matching))
            f.write('\nRatio bodies over all positives: ' + f"{ratio_bodies_over_all_positives:.3f}")
        
    
if __name__ == "__main__":
    main()
