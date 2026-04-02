#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################   PARAMETERS   #############################

oir_folder      = '' # Write the folder path, without the last \ or / . In Windows, place an r before the ''.
folder_models   = '' # Write the folder path, without the last \ or / . In Windows, place an r before the ''.

# Change for every folder and microscope set up:

signal_of_interest = 'PRPF6'
list_speckle_markers = ['PRPF4B'] #This has to be a list!
list_classes = ['WT', 'H100Afs', 'M725T', 'M639V', 'A589P', 'D670N'] #This has to be a list!
nuclei_channel_number = 0
speckle_channel_number = 1
signal_channel_number = 3

###########################################################################

# Rarely you will chenge this:

ending_volume_speckle = '_volume_speckle_channel.tiff'
ending_volume_nuclei = '_volume_nuclei_channel.tiff'
ending_volume_PRPF6 = '_volume_'+signal_of_interest+'_channel.tiff'

ending_volume_nuclei_segmentation = '_nuclei_segmentation.tiff'
ending_volume_speckle_segmentation = '_speckle_segmentation.tiff'
ending_volume_around_speckle_segmentation = '_around_speckle_segmentation.tiff'
ending_volume_nuclei_segmentation_binary = '_nuclei_segmentation_binary.tiff'
ending_volume_speckle_segmentation_binary = '_speckle_segmentation_binary.tiff'
ending_volume_around_speckle_segmentation_binary = '_around_speckle_segmentation_binary.tiff'

model_name_nuclei   =  'nuclei_60x_model_nuclei_diam_100_ji_0.9638.634204'
model_name_speckle  =  'Speckle_SC35_4B_60x_model_nuclei_diam_28_ji_0.5785.073982'

flag_gpu = False # If not sure, use False
flag_closing = False
flag_filter_by_size = True
flag_create_folders = True
min_pixels_matching = 100
list_distance_expansion = [3]
list_threshold = [300]
list_flag_use_median = [True]

###########################################################################

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)


from quantify_segmentation import get_expr_from_labels, matching_label_pairs, generate_props_dict
from aux_functions.functions_3D import segment_slice_by_slice, get_props_per_cell3D
import os
from aux_functions.template_paraview_python_script import TEMPLATE_HEAD
from aux_functions.functionSaveTIFFMultipage import functionSaveTIFFMultipage
from skimage.segmentation import expand_labels
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aux_functions.functionPercNorm import functionPercNorm

import os

def get_volume_from_ome_channels(oir_file, flag_norm=True):

    base_dir  = os.path.dirname(oir_file)
    base_name = os.path.splitext(os.path.basename(oir_file))[0]

    # Detect channel files
    channel_files = []
    ch = 0
    while True:
        fname = os.path.join(base_dir, f"{base_name}_CH{ch}.ome.tif")
        if os.path.exists(fname):
            channel_files.append(fname)
            ch += 1
        else:
            break

    if len(channel_files) == 0:
        raise FileNotFoundError(f"No channel OME-TIFF files found for {oir_file}")

    print(f"Found {len(channel_files)} channels")

    list_channels = []
    voxel_size_x = voxel_size_y = voxel_size_z = None

    for ch_idx, ch_file in enumerate(channel_files):

        img = AICSImage(ch_file)
        # data = img.get_image_data("TCZYX")
        data = img.get_image_data("TCXYZ")
        dims = img.dims

        size_t = dims.T
        size_z = dims.Z
        size_y = dims.Y
        size_x = dims.X

        if ch_idx == 0:
            voxel_size_x = img.physical_pixel_sizes.X
            voxel_size_y = img.physical_pixel_sizes.Y
            voxel_size_z = img.physical_pixel_sizes.Z

            #print(f"SizeT: {size_t}")
            #print(f"SizeZ: {size_z}")
            #print(f"SizeX: {size_x}")
            #print(f"SizeY: {size_y}")
            #print(f"VoxelSizeX: {voxel_size_x}")
            #print(f"VoxelSizeY: {voxel_size_y}")
            #print(f"VoxelSizeZ: {voxel_size_z}")

        #channel_volume = np.zeros((size_y, size_x, size_z), dtype=data.dtype)
        channel_volume = np.zeros((size_x, size_y, size_z))

        for z in range(size_z):
            #image = data[0, 0, z, :, :]  # T=0, C=0
            image = data[0, 0, :, :, z]  # T=0, C=0
            if flag_norm:
                channel_volume[:, :, z] = functionPercNorm(image)
            else:
                channel_volume[:, :, z] = image

        list_channels.append(channel_volume)

    return list_channels, voxel_size_x, voxel_size_y, voxel_size_z

def main():
    fullpath_model_speckle = os.path.join(folder_models, model_name_speckle)
    fullpath_model_nuclei = os.path.join(folder_models, model_name_nuclei)
    if not (os.path.exists(fullpath_model_nuclei)) and not(os.path.exists(fullpath_model_speckle)):
        print("Nuclei model does not exist.")
        sys.exit(1)
    
    
    print("------START MAIN--------")
    
    try:
        
        txt_output = ''
        output_txt = os.path.join(oir_folder,'experiment_output.txt')
        
        for flag_use_median in list_flag_use_median:
            for size_min in list_threshold:
                for distance_expansion in list_distance_expansion:
                
                    ending_file = '_matching_'+str(min_pixels_matching) + '_closing_' + str(flag_closing) + \
                        '_expansion_' + str(distance_expansion) + '_median_' + str(flag_use_median) + \
                            '_filterBySize_' + str(flag_filter_by_size) + '_' + str(size_min)
                    
                    list_all_filename_image = []
                    # list_all_nuclei_expression = []
                    list_all_around_speckle_expression = []
                    list_all_outside_speckle_expression = []
                    list_all_speckle_expression = []
                    list_all_ratio_in_around_speckle_expression = []
                    list_all_ratio_in_out_speckle_expression = []
                    list_all_label_image = []
                    list_all_speckle_marker = []
                    list_all_signal_channel_number = []
                    list_all_speckle_channel_number = []
                    list_all_nuclei_id = []
                    list_all_speckle_id = []
                    list_all_speckle_voxels = []
                    
                    list_mean_around_speckle_expression = []
                    list_mean_speckle_expression = []
                    list_mean_outside_expression = []
                    list_label_image = []

                    files_tiff = []
                    list_classes_effective = []
                    for file in os.listdir(oir_folder):
                        if file.endswith(".oir"):
                            files_tiff.append(file)
                    
                    for oir_file in files_tiff:
                        
                        fullpath_oir = os.path.join(oir_folder, oir_file)
                        
                        label_image = next((x for x in list_classes if x in oir_file), None)
                        
                        speckle_marker = next((x for x in list_speckle_markers if x in oir_file), None)
                        
                        if (label_image is not None) and (speckle_marker is not None):
                            if not(label_image in list_classes_effective):
                                list_classes_effective.append(label_image)
                            folder_output_sample = fullpath_oir + ending_file
                            if flag_create_folders:
                                if not os.path.exists(folder_output_sample):
                                    os.makedirs(folder_output_sample)
                            
                            list_volumes, voxel_size_x, voxel_size_y, voxel_size_z = get_volume_from_ome_channels(fullpath_oir, flag_norm = True)
                            list_images = list_volumes
                            n_images = len(list_images)
                            print('n volumes ', n_images)
                            
                            channel_nuclei = list_images[nuclei_channel_number]
                            channel_nuclei = channel_nuclei * 255.
                            
                            channel_speckle = list_images[speckle_channel_number]
                            channel_speckle = channel_speckle * 255.
                            
                            channel_PRPF6 = list_images[signal_channel_number]
                            channel_PRPF6 = channel_PRPF6 * 255.
                            
                            
                            physical_x_y = 1.0
                            ratio_z = voxel_size_z / voxel_size_y
                            
                            # Saving volumes
                            print('Saving volumes')
                            
                            fullpath_tiff_channel_speckle = os.path.join(folder_output_sample,oir_file + ending_volume_speckle)
                            if flag_create_folders:
                                functionSaveTIFFMultipage(channel_speckle, fullpath_tiff_channel_speckle, 8)
                            
                            fullpath_tiff_channel_nuclei = os.path.join(folder_output_sample,oir_file + ending_volume_nuclei)
                            if flag_create_folders:
                                functionSaveTIFFMultipage(channel_nuclei, fullpath_tiff_channel_nuclei, 8)
                            
                            fullpath_tiff_channel_PRPF6 = os.path.join(folder_output_sample,oir_file + ending_volume_PRPF6)
                            if flag_create_folders:
                                functionSaveTIFFMultipage(channel_PRPF6, fullpath_tiff_channel_PRPF6, 8)
                            
                            # Segment volumes
                            print('Segmenting volumes')
                            masks_nuclei = segment_slice_by_slice(channel_nuclei, fullpath_model_nuclei, diameter=None, flag_gpu = flag_gpu, \
                                                                  flag_closing = flag_closing, flag_filter_by_size = flag_filter_by_size, size_min = size_min)
                            total_nuclei = len(np.unique(masks_nuclei)) - 1
                            fullpath_tiff_nuclei_segmentation = os.path.join(folder_output_sample,oir_file + ending_volume_nuclei_segmentation)
                            fullpath_tiff_nuclei_segmentation_binary = os.path.join(folder_output_sample,oir_file + ending_volume_nuclei_segmentation_binary)
                            if flag_create_folders:
                                functionSaveTIFFMultipage(masks_nuclei, fullpath_tiff_nuclei_segmentation, 8)
                                functionSaveTIFFMultipage(np.uint8(masks_nuclei>0)*255, fullpath_tiff_nuclei_segmentation_binary, 8)
                            
                            
                            print('Segmenting speckle')
                            mask_speckle = segment_slice_by_slice(channel_speckle, fullpath_model_speckle, diameter=None, flag_gpu = flag_gpu, \
                                                                  flag_closing = flag_closing , flag_filter_by_size = flag_filter_by_size, size_min = size_min)
                            total_speckle = len(np.unique(mask_speckle)) - 1
                            # Around speckles
                            mask_speckle_around = expand_labels(mask_speckle, distance = distance_expansion)
                            mask_speckle_around[mask_speckle>0] = 0 # Empty in the speckle area
                            
                            # total_mito = len(np.unique(mask_mito)) - 1
                            fullpath_tiff_speckle_segmentation = os.path.join(folder_output_sample,oir_file + ending_volume_speckle_segmentation)
                            fullpath_tiff_speckle_segmentation_binary = os.path.join(folder_output_sample,oir_file + ending_volume_speckle_segmentation_binary)
                            if flag_create_folders:
                                functionSaveTIFFMultipage(mask_speckle, fullpath_tiff_speckle_segmentation, 8)
                                functionSaveTIFFMultipage(np.uint8(mask_speckle>0)*255, fullpath_tiff_speckle_segmentation_binary, 8)
                            
                            fullpath_tiff_around_speckle_segmentation = os.path.join(folder_output_sample,oir_file + ending_volume_around_speckle_segmentation)
                            fullpath_tiff_around_speckle_segmentation_binary = os.path.join(folder_output_sample,oir_file + ending_volume_around_speckle_segmentation_binary)
                            if flag_create_folders:
                                functionSaveTIFFMultipage(mask_speckle_around, fullpath_tiff_around_speckle_segmentation, 8)
                                functionSaveTIFFMultipage(np.uint8(mask_speckle_around>0)*255, fullpath_tiff_around_speckle_segmentation_binary, 8)
                            
                            # Only segmented speckles inside the segmented nuclei
                            props_speckles = get_props_per_cell3D(mask_speckle)
                            props_speckles_dict = generate_props_dict(props_speckles)
                            _, _, matching_pairs_a_to_b_non_zero = matching_label_pairs(masks_nuclei, mask_speckle, min_pixels=min_pixels_matching)
                            
                            if len(matching_pairs_a_to_b_non_zero)>0:
                                left, right = zip(*matching_pairs_a_to_b_non_zero)
                                labels_nuclei_matched = list(left)
                                labels_speckle_matched = list(right)
                                list_speckle_expression = get_expr_from_labels(labels_speckle_matched, mask_speckle, channel_PRPF6, flag_use_median = flag_use_median)
                                list_around_speckle_expression = get_expr_from_labels(labels_speckle_matched, mask_speckle_around, channel_PRPF6, flag_use_median = flag_use_median)
                                
                                list_size_speckle = []
                                for label_speckle in labels_speckle_matched:
                                    list_size_speckle.append((props_speckles_dict[label_speckle]).volume)
                                # Expression outside the speckles
                                masks_nuclei_not_speckle = masks_nuclei.copy()
                                masks_nuclei_not_speckle[mask_speckle>0] = 0
                                masks_nuclei_not_speckle[mask_speckle_around>0] = 0
                                list_expression_outside = get_expr_from_labels(labels_nuclei_matched, masks_nuclei_not_speckle, channel_PRPF6, flag_use_median = flag_use_median)
                                
                                list_all_nuclei_id.extend(labels_nuclei_matched)
                                list_all_speckle_id.extend(labels_speckle_matched)
                                list_all_speckle_expression.extend(list_speckle_expression)
                                list_all_speckle_voxels.extend(list_size_speckle)
                                
                                list_all_around_speckle_expression.extend(list_around_speckle_expression)
                                list_all_outside_speckle_expression.extend(list_expression_outside)
                                
                                ratio_in_around = np.divide(np.array(list_around_speckle_expression),np.array(list_speckle_expression))
                                ratio_in_outside = np.divide(np.array(list_expression_outside),np.array(list_speckle_expression))
                                list_all_ratio_in_around_speckle_expression.extend(list(ratio_in_around))
                                list_all_ratio_in_out_speckle_expression.extend(list(ratio_in_outside))
                                
                                list_all_label_image.extend([label_image] * len(list_speckle_expression))
                                list_all_filename_image.extend([oir_file] * len(list_speckle_expression))
                                list_all_speckle_marker.extend([speckle_marker] * len(list_speckle_expression))
                                list_all_signal_channel_number.extend([signal_channel_number] * len(list_speckle_expression))
                                list_all_speckle_channel_number.extend([speckle_channel_number] * len(list_speckle_expression))
                                
                                list_mean_speckle_expression.append(np.mean(np.array(list_speckle_expression)))
                                list_mean_around_speckle_expression.append(np.mean(np.array(list_around_speckle_expression)))
                                list_mean_outside_expression.append(np.mean(np.array(list_expression_outside)))
                                list_label_image.append(label_image)
                                
                                # Creating script for Paraview
                                print('Creating script for Paraview')
                                
                                colormap_channel_nuclei = "Cool to Warm"
                                colormap_channel_speckle = "Viridis (matplotlib)"
                                colormap_channel_SC35 = "Blue - Green - Orange"
                                colormap_nuclei_segmentation = "X Ray"
                                colormap_speckle_segmentation = "Rainbow Desaturated"
                                colormap_around_speckle_segmentation = "Rainbow Desaturated"
                                volume_list = [
                                (fullpath_tiff_channel_nuclei, 'channel_nuclei', colormap_channel_nuclei),
                                (fullpath_tiff_channel_speckle, 'channel_speckle', colormap_channel_speckle),
                                (fullpath_tiff_channel_PRPF6, 'channel_'+signal_of_interest, colormap_channel_SC35),
                                (fullpath_tiff_nuclei_segmentation, 'segmentation_nuclei', colormap_nuclei_segmentation),
                                (fullpath_tiff_speckle_segmentation, 'segmentation_speckle', colormap_speckle_segmentation),
                                (fullpath_tiff_around_speckle_segmentation, 'segmentation_speckle', colormap_around_speckle_segmentation)
                                ]
                                
                                output_script_path = os.path.join(folder_output_sample, oir_file + "_render_volumes.py")
                                
                                load_calls = ""
                                for tiff_path, registration_name, colormap in volume_list:
                                    tiff_escaped = tiff_path.replace("\\", "\\\\")  # escape backslashes for Windows
                                    load_calls += f"load_volume('{tiff_escaped}', '{registration_name}', '{colormap}', {physical_x_y}, {physical_x_y}, {ratio_z})\n"
                                
                                # Combine all parts
                                final_script = TEMPLATE_HEAD + load_calls
                                
                                # Write the generated script to a file
                                if flag_create_folders:
                                    with open(output_script_path, "w") as f:
                                        f.write(final_script)
                                    
                                if flag_create_folders:    
                                    txt_output_sample = os.path.join(folder_output_sample, oir_file + '_summary.txt')
                                    f = open(txt_output_sample, "w")
                                    
                                    f.write(oir_file + '\n')
                                    f.write('------------------------------------------------------- \n')        
                                    f.write('N channels: ' + str(n_images) + '\n')
                                    f.write('Channels (starting in 0)\n')
                                    f.write('Channel speckles: ' + str(speckle_channel_number) + '\n')
                                    f.write('Channel signal: ' + str(signal_channel_number) + '\n')
                                    
                                    f.write('------------------------------------------------------- \n')        
                                    f.write('Total speckles: ' + str(total_speckle) + '\n')
                                    f.write('Total valid speckles: ' + str( len(labels_speckle_matched)) + '\n')
                                    f.write('Total nuclei: ' + str( total_nuclei) + '\n')
                                    f.write('------------------------------------------------------- \n')        
                                    f.write('physical_x_y: ' + str( physical_x_y) + '\n')
                                    f.write('ratio_z: ' + str( ratio_z) + '\n')
                                    
                                    f.close()
                                    del f
                                
                        else:
                            print(oir_file," - NO nuclei or speckle found")
                    
                    # str_mean_median = 'median' if flag_use_median else 'mean'
                    df = pd.DataFrame({
                        'Image': list_all_filename_image,
                        'Treatment': list_all_label_image,
                        'Speckle_marker': list_all_speckle_marker,
                        'Speckle_channel': list_all_speckle_channel_number,
                        signal_of_interest + '_channel': list_all_signal_channel_number,
                        'Nuclei_id': list_all_nuclei_id,
                        'Speckle_id': list_all_speckle_id,
                        'Speckle_size': list_all_speckle_voxels,
                        'Expression_in_speckle': list_all_speckle_expression,
                        'Expression_around_speckle': list_all_around_speckle_expression,
                        'Expression_outside_speckle': list_all_outside_speckle_expression,
                        'Ratio_in_around': list_all_ratio_in_around_speckle_expression,
                        'Ratio_in_out': list_all_ratio_in_out_speckle_expression
                    })
                    
                    # save to csv
                    csv_output_expressions = os.path.join(oir_folder, 'SPECKLE-LEVEL-expressions'+ ending_file +'.csv')
                    df.to_csv(csv_output_expressions, index=False)
                    
        
        with open(output_txt, "w") as f:
            f.write(txt_output)
        
        
    finally:
        print("------END--------")

    
if __name__ == "__main__":
    main()
