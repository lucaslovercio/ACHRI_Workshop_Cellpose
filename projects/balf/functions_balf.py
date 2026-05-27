#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:55:34 2026

@author: lucas
"""

import joblib
import cv2
from cellpose import models
from quantify_segmentation import get_props_per_cell, get_labels, matching_label_pairs, get_compactness
from aux_functions import functionPercNorm
import numpy as np
from torchvision import transforms
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes

eps_div = 0.0000001

CELL_TYPES = {
    'e': ('Eosinophil',  np.uint8(1)),
    'm': ('Macrophage',  np.uint8(2)),
    'r': ('RBC',         np.uint8(3)),
    'o': ('Monocyte',    np.uint8(4)),
    'l': ('Lymphocyte',  np.uint8(5)),
    'n': ('Neutrophil',  np.uint8(6)),
    'p': ('Epithelial',  np.uint8(7)),
    'j': ('Junk',        np.uint8(8)),
    'b': ('Background',  np.uint8(0)),
}

def label_to_name(label):
    return CELL_TYPES[label][0] if label in CELL_TYPES else label

def get_channels_histology_normalized(image_to_transform, normalizer, flag_hist_normalize):
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
    ])
    
    t_to_transform = T(image_to_transform)
    result_normalization  = normalizer.normalize(I=t_to_transform)
    img_rgb = result_normalization[0] if isinstance(result_normalization, tuple) else result_normalization
    img_rgb = np.uint8(img_rgb.numpy())
    
    
    channel_r_nuclei = np.uint8(img_rgb[:,:,0])
    channel_g_cell = np.uint8(img_rgb[:,:,1])
    channel_b_cell = np.uint8(img_rgb[:,:,2])

    if flag_hist_normalize:
        channel_r_nuclei = functionPercNorm( np.single(channel_r_nuclei))
        channel_g_cell = functionPercNorm( np.single(channel_g_cell))
        channel_b_cell = functionPercNorm( np.single(channel_b_cell))
    else:
        channel_r_nuclei = np.single(channel_r_nuclei) / 255.
        channel_g_cell = np.single(channel_g_cell) / 255.
        channel_b_cell = np.single(channel_b_cell) / 255.
        
    return img_rgb, channel_r_nuclei, channel_g_cell, channel_b_cell

def segment_image_normalizer(full_path, normalizer, path_model_cells, path_model_nuclei,
                  flag_gpu, channels = [[0, 0]]):
    img_rgb_original = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)

    img_rgb, channel_r_nuclei, channel_g_cell, channel_b_cell = \
        get_channels_histology_normalized(img_rgb_original, normalizer, flag_hist_normalize=False)

    # Segment cells on normalized image
    model_trained_cells = models.CellposeModel(pretrained_model=path_model_cells, gpu=flag_gpu)
    img_segmentation_cells, _, _ = model_trained_cells.eval(channel_g_cell, diameter=None, channels=channels)
    del model_trained_cells

    # Segment nuclei on normalized image
    model_trained_nuclei = models.CellposeModel(pretrained_model=path_model_nuclei, gpu=flag_gpu)
    img_segmentation_nuclei, _, _ = model_trained_nuclei.eval(channel_r_nuclei, diameter=None, channels=channels)
    del model_trained_nuclei

    return img_rgb, img_rgb_original, channel_r_nuclei, channel_g_cell, channel_b_cell, \
           img_segmentation_cells, img_segmentation_nuclei

def get_features_matrix_selectedSVM(valid_cells_id, img_segmentation_cells, vector_px_nuclei_in_cell, counter_occurrence_cell_lobes, \
                        vector_mean_r_in_cell, vector_mean_g_in_cell, vector_mean_b_in_cell, \
                            vector_mean_r_in_cyto, vector_mean_g_in_cyto, vector_mean_b_in_cyto, \
                                vector_median_r_in_cyto, vector_median_g_in_cyto, vector_median_b_in_cyto, \
                                eccentricity_nucleus, compactness_nucleus,\
                                solidity_nucleus, has_hole_nucleus):
        
    features_names = ['G_cyto_norm', 'R_cyto_norm', 'R_over_B', 'R_over_G', 'area',\
                      'compactness', 'compactness_nucleus', 'eccentricity',\
                          'mean_r_in_cyto', 'median_g_in_cyto', 'median_r_in_cyto',\
                          'n_cell_lobes', 'perimeter', 'px_nuclei_in_cell',\
                              'ratio_b_cell', 'ratio_g_cell', 'ratio_nucleus_cell', 'ratio_r_cell',\
                              'total_cyto_bright']
    
    matrix_features_cells = []
    cells_id_added = []

    props_cells = get_props_per_cell(img_segmentation_cells) #sorted by label
    
    pos_cell = 0
    for cell in props_cells:
        
        if cell.label in valid_cells_id:
            cell_id_pos = pos_cell# valid_cells_id.index(cell.label)

            ratio_nucleus_cell = np.float32(vector_px_nuclei_in_cell[cell_id_pos]) / np.float32(cell.area + eps_div)
            ratio_r_cell = np.float32(vector_mean_r_in_cell[cell_id_pos])
            ratio_g_cell = np.float32(vector_mean_g_in_cell[cell_id_pos])
            ratio_b_cell = np.float32(vector_mean_b_in_cell[cell_id_pos])
            
            # Normalization of color in cytoplasm
            total_cyto_bright = np.float32(vector_mean_r_in_cyto[cell_id_pos] + vector_mean_g_in_cyto[cell_id_pos] + vector_mean_b_in_cyto[cell_id_pos]) + eps_div
            R_cyto_norm = vector_mean_r_in_cyto[cell_id_pos] / total_cyto_bright
            G_cyto_norm = vector_mean_g_in_cyto[cell_id_pos] / total_cyto_bright
            B_cyto_norm = vector_mean_b_in_cyto[cell_id_pos] / total_cyto_bright
            R_over_G = R_cyto_norm / (G_cyto_norm + eps_div)
            R_over_B = R_cyto_norm / (B_cyto_norm + eps_div)
            
            row_temp = [G_cyto_norm, R_cyto_norm, R_over_B, R_over_G, np.float32(cell.area),\
                        np.float32(cell.compactness), np.float32(compactness_nucleus[cell_id_pos]), np.float32(cell.eccentricity),\
                        vector_mean_r_in_cyto[cell_id_pos], vector_median_g_in_cyto[cell_id_pos], vector_median_r_in_cyto[cell_id_pos],\
                            np.float32(counter_occurrence_cell_lobes[cell_id_pos]), np.float32(cell.perimeter), np.float32(vector_px_nuclei_in_cell[cell_id_pos]),\
                                ratio_b_cell, ratio_g_cell, ratio_nucleus_cell, ratio_r_cell,\
                                    total_cyto_bright]
                
            if len(row_temp) != len(features_names):
                raise ValueError(f"Length mismatch: {len(row_temp)} vs {len(features_names)}")
                
            matrix_features_cells.append(row_temp)
            cells_id_added.append(cell.label)
        pos_cell = pos_cell + 1
        
    return matrix_features_cells, cells_id_added, features_names

def get_mean_intensity_per_cell(cells_id, img_segmentation_cells, channel):
    vector_intensity_nuclei_in_cell = np.float32(np.zeros_like(cells_id))
    vector_intensity_mean_nuclei_in_cell = np.float32(np.zeros_like(cells_id))
    for i in range(len(cells_id)):
        cell_id = cells_id[i]
        if cell_id>0: # Do not count in background
            px_nuclei_in_cell = np.sum(np.float32(channel[img_segmentation_cells==cell_id]))
            vector_intensity_nuclei_in_cell[i] = px_nuclei_in_cell
            px_count = np.sum(np.float32(img_segmentation_cells==cell_id))
            mean_intensity = px_nuclei_in_cell / (px_count + eps_div)
            vector_intensity_mean_nuclei_in_cell[i] = mean_intensity
        
    return vector_intensity_nuclei_in_cell, vector_intensity_mean_nuclei_in_cell

def get_intensities_in_cyto(cells_id, img_segmentation_cells, img_segmentation_nuclei, channel, flag_use_median = False):
   vector_mean_intensity_in_cito = np.float32(np.zeros_like(cells_id))
   for i in range(len(cells_id)):
       cell_id = cells_id[i]
       if cell_id>0: # Do not count in background
           mask = np.logical_and(img_segmentation_cells==cell_id,img_segmentation_nuclei==0) # all cell minus the pixels in the nuclei
           colours_px = np.float32(channel[mask])
           if len(colours_px)>0:
               if flag_use_median:
                   vector_mean_intensity_in_cito[i] = np.median(colours_px)
               else:
                   vector_mean_intensity_in_cito[i] = np.mean(colours_px)
       
   return vector_mean_intensity_in_cito

def get_geometry_nuclei(img_segmentation_cells, img_segmentation_nuclei, props_cells, matching_pairs_non_zero_cells_nuclei, margin_bbox = 2):
    results = []
    h,w = img_segmentation_cells.shape
    for cell in props_cells:

        start_r, start_c, end_r, end_c = cell.bbox
        start_r = max([start_r-margin_bbox,0])
        start_c = max([start_c-margin_bbox,0])
        end_r = min([end_r+margin_bbox,h])
        end_c = min([end_c+margin_bbox,w])
        # Crop both segmentations
        crop_nuclei = np.copy(img_segmentation_nuclei[start_r:end_r, start_c:end_c])
        
        for pair_c_n in matching_pairs_non_zero_cells_nuclei:
            if cell.label != pair_c_n[0]:
                crop_nuclei[crop_nuclei==pair_c_n[1]] = 0 # Erase any nuclei label not belonging to the cell
        
        crop_cell = np.copy(img_segmentation_cells[start_r:end_r, start_c:end_c])
        crop_cell = crop_cell == cell.label
        crop_nuclei_binary = np.logical_and(crop_nuclei > 0, crop_cell > 0)
        
        # Binarize the result
        binary_mask = crop_nuclei_binary # dilated_nuclei.astype(np.uint8)

        # Label the binary region (in case there are multiple connected parts)
        labeled = label(binary_mask)

        # binary image: True = object, False = background
        filled = binary_fill_holes(binary_mask)
        has_hole = np.any(filled & ~binary_mask)
        
        region_props = regionprops(labeled)
        if len(region_props) >0:
            region = region_props[0] #I should not have more than 1
            eccentricity = region.eccentricity
            solidity = region.solidity

            area = region.area
            perimeter = region.perimeter + eps_div  # avoid divide-by-zero
            compactness = get_compactness(area,perimeter)
            
            results.append([cell.label, area, perimeter, eccentricity, compactness, solidity, has_hole])

    return results

def get_lobe_data(img_segmentation_cells, img_segmentation_nuclei, min_pixels_matching = 25):
    props_cells = get_props_per_cell(img_segmentation_cells)
    
    matching_pairs_cells_nuclei, matching_pairs_non_zero_left, matching_pairs_non_zero_cells_nuclei = \
        matching_label_pairs(img_segmentation_cells, img_segmentation_nuclei, min_pixels=min_pixels_matching)
        
    # How many nucleus lobes has each cell?
    cells_id = get_labels(props_cells)
    counter_occurrence_cell_lobes= np.zeros_like(cells_id)
    eccentricity_nucleus= np.float32(np.zeros_like(cells_id))
    compactness_nucleus= np.float32(np.zeros_like(cells_id))
    solidity_nucleus = np.float32(np.zeros_like(cells_id))
    has_hole_nucleus = np.float32(np.zeros_like(cells_id))
    img_lobes_count = np.zeros_like(img_segmentation_cells)

    for pair_c_n in matching_pairs_non_zero_cells_nuclei:
        cell_id_temp = pair_c_n[0]

        pos_cell_id = cells_id.index(cell_id_temp)

        counter_occurrence_cell_lobes[pos_cell_id] = counter_occurrence_cell_lobes[pos_cell_id] + 1
        img_lobes_count[img_segmentation_cells==cell_id_temp] = counter_occurrence_cell_lobes[pos_cell_id]
    
    geometries_nuclei = get_geometry_nuclei(img_segmentation_cells, img_segmentation_nuclei, props_cells, matching_pairs_non_zero_cells_nuclei)

    for geometry_nucleus in geometries_nuclei:
        # print(geometry_nucleus)
        cell_id_temp = geometry_nucleus[0]
        pos_cell_id = cells_id.index(cell_id_temp)
        eccentricity_nucleus[pos_cell_id] = geometry_nucleus[3]
        compactness_nucleus[pos_cell_id] = geometry_nucleus[4]
        solidity_nucleus[pos_cell_id] = geometry_nucleus[5]
        has_hole_nucleus[pos_cell_id] = np.float32(geometry_nucleus[6])
    return cells_id, counter_occurrence_cell_lobes, img_lobes_count, eccentricity_nucleus, compactness_nucleus, solidity_nucleus, has_hole_nucleus

def get_px_nuclei_per_cell(cells_id, img_segmentation_cells, img_segmentation_nuclei):
    vector_px_nuclei_in_cell = np.zeros_like(cells_id)
    vector_proportion_nuclei_in_cell = np.float32(np.zeros_like(cells_id))
    for i in range(len(cells_id)):
        cell_id = cells_id[i]
        if cell_id>0: # Do not count in background
            px_nuclei_in_cell = np.sum(img_segmentation_nuclei[img_segmentation_cells==cell_id]>0) #Number of pixels of nuclei
            px_cell = np.sum((img_segmentation_cells==cell_id)>0) #Total pixels in cell
            vector_px_nuclei_in_cell[i] = px_nuclei_in_cell
            vector_proportion_nuclei_in_cell[i] = np.float32(px_nuclei_in_cell) / np.float32(px_cell)
        
    return vector_px_nuclei_in_cell, vector_proportion_nuclei_in_cell

def get_features_table_images(img_segmentation_cells_image, img_segmentation_nuclei_image, valid_labels_cells,\
                              channel_r_nuclei_image, channel_g_cell_image, channel_b_cell_image, min_pixels_matching):
    
    labels_cells_image, counter_occurrence_cell_lobes_image, img_lobes_count_image, eccentricity_nucleus_image, compactness_nucleus_image, solidity_nucleus_image, has_hole_nucleus_image = \
        get_lobe_data(img_segmentation_cells_image, img_segmentation_nuclei_image, min_pixels_matching = min_pixels_matching)
    
    vector_px_nuclei_in_cell_image, _ = get_px_nuclei_per_cell(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image)
    
    _, vector_mean_r_in_cell_image = get_mean_intensity_per_cell(labels_cells_image, img_segmentation_cells_image, channel_r_nuclei_image)
    _, vector_mean_g_in_cell_image = get_mean_intensity_per_cell(labels_cells_image, img_segmentation_cells_image, channel_g_cell_image)
    _, vector_mean_b_in_cell_image = get_mean_intensity_per_cell(labels_cells_image, img_segmentation_cells_image, channel_b_cell_image)
    
    vector_mean_r_in_cyto_image = get_intensities_in_cyto(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image, channel_r_nuclei_image)
    vector_mean_g_in_cyto_image = get_intensities_in_cyto(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image, channel_g_cell_image)
    vector_mean_b_in_cyto_image = get_intensities_in_cyto(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image, channel_b_cell_image)
    
    vector_median_r_in_cyto_image = get_intensities_in_cyto(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image, channel_r_nuclei_image, flag_use_median=True)
    vector_median_g_in_cyto_image = get_intensities_in_cyto(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image, channel_g_cell_image, flag_use_median=True)
    vector_median_b_in_cyto_image = get_intensities_in_cyto(labels_cells_image, img_segmentation_cells_image, img_segmentation_nuclei_image, channel_b_cell_image, flag_use_median=True)
    
    matrix_features_cells_image, cells_id_added, features_names = get_features_matrix_selectedSVM(valid_labels_cells, img_segmentation_cells_image, vector_px_nuclei_in_cell_image, counter_occurrence_cell_lobes_image, \
                                                    vector_mean_r_in_cell_image, vector_mean_g_in_cell_image, vector_mean_b_in_cell_image, \
                                                        vector_mean_r_in_cyto_image, vector_mean_g_in_cyto_image, vector_mean_b_in_cyto_image, \
                                                            vector_median_r_in_cyto_image, vector_median_g_in_cyto_image, vector_median_b_in_cyto_image, \
                                                        eccentricity_nucleus_image, compactness_nucleus_image, solidity_nucleus_image, has_hole_nucleus_image)
    
    return matrix_features_cells_image, cells_id_added, features_names

def process_image(input_image, path_model_cells, path_model_nuclei,
                  path_rf_celltypes, path_transformer_reinhard, min_pixels_matching = 200, flag_gpu = False):
    
    normalizer = joblib.load(path_transformer_reinhard)
    img_rgb, img_rgb_original, channel_r_nuclei, channel_g_cell, channel_b_cell, \
        img_segmentation_cells, img_segmentation_nuclei = \
        segment_image_normalizer(input_image, normalizer, path_model_cells, path_model_nuclei, flag_gpu)
    
    # --- Feature extraction ---
    props_cells = get_props_per_cell(img_segmentation_cells)
        
    # How many nucleus lobes has each cell?
    cells_id = get_labels(props_cells)
    n_cells = len(cells_id)

    matrix_features_cells_temp, _ , descriptors_name = get_features_table_images(img_segmentation_cells, img_segmentation_nuclei, cells_id,\
                                                           channel_r_nuclei, channel_g_cell, channel_b_cell, min_pixels_matching)

    # --- Classification ---
    rf_classifier   = joblib.load(path_rf_celltypes)
    
    if len(matrix_features_cells_temp) == 0:
        print('No cells found in this tile, skipping classification.')
        pred_str =[] 
        pred_confidence = []
        del rf_classifier, matrix_features_cells_temp
    else:
        try:
            pred_temp       = rf_classifier.predict(matrix_features_cells_temp)
            proba_temp      = rf_classifier.predict_proba(matrix_features_cells_temp)
            pred_str        = list(pred_temp)
            pred_confidence = list(proba_temp.max(axis=1))
        except ValueError as e:
            print(f'  Warning: classification skipped due to invalid feature values: {e}')
            pred_str, pred_confidence, cells_id = [], [], []
            n_cells = 0
        finally:
            del rf_classifier, matrix_features_cells_temp
        
    return (img_rgb, img_rgb_original, img_segmentation_cells, img_segmentation_nuclei,
            n_cells, pred_str, pred_confidence, cells_id)