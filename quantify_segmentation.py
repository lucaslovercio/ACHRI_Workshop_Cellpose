#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:02:49 2023

@author: lucas
"""

from skimage import measure
import os
import matplotlib.pyplot as plt
from cellpose.io import imread
import numpy as np
import csv
import pandas

###################################   PARAMETERS   #########################

folder_original = ''
folder_segmentations = ''
ending_segmentation = '_cellpose.png'
flag_show = True
th_size = 10000

##############################################################################

class CellProperty:
  def __init__(self, xCentroid, yCentroid,label, area, perimeter, eccentricity, compactness, bbox):
    self.xCentroid = xCentroid
    self.yCentroid = yCentroid
    self.label = label
    self.area = area
    self.perimeter = perimeter
    self.eccentricity = eccentricity
    self.compactness = compactness
    self.bbox = bbox
  
  def to_dict(self):
    return {
        'xCentroid': self.xCentroid,
        'yCentroid': self.yCentroid,
        'label': self.label,
        'area': self.area,
        'perimeter': self.perimeter,
        'eccentricity': self.eccentricity,
        'compactness': self.compactness,
    }

def get_compactness(area, perimeter):
    compactness = (4. * np.pi * area.astype('float')) / (perimeter.astype('float') * perimeter.astype('float') + 0.0000001)
    return compactness

def get_props_per_cell(img_segmentation):
    
    regions = measure.regionprops(img_segmentation)
    regionprops_selected = []
    for region in regions:
        
        binary_image = np.where(img_segmentation == region.label,1,0)
        perimeter = measure.perimeter(binary_image, neighborhood=8)
        compactness = get_compactness(region.area, region.perimeter)
        #y is first dimension, x is second dimension
        cell_property = CellProperty(region.centroid[1], region.centroid[0], region.label, region.area, perimeter, region.eccentricity, compactness, region.bbox)
        regionprops_selected.append(cell_property)
    
    return regionprops_selected
    
def detect_big_cells(cell_props, th_size=9000):
    vector_area_per_cell = []
    vector_idx = []
    for cell_prop in cell_props:
        vector_area_per_cell.append(cell_prop.area)
        vector_idx.append(cell_prop.label)
    
    #vector_idx = vector_idx.array()
    big_cells = np.asarray(vector_area_per_cell) > th_size
    big_cells = np.nonzero(big_cells)
    big_cells_idx = np.take(vector_idx,big_cells)
    big_cells_idx = big_cells_idx[0] #+ 1
    
    return vector_area_per_cell, big_cells_idx

def list_cells(cell_props, csv_output):
    with open(csv_output, 'w', newline='') as csvfile:
        fieldnames = ['Label', 'xCentroid', 'yCentroid', 'Area', 'Perimeter', 'Eccentricity', 'Compactness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        for cell_prop in cell_props:
            writer.writerow({'Label': str(cell_prop.label), 'xCentroid': str(cell_prop.xCentroid), 'yCentroid': str(cell_prop.yCentroid),\
                             'Area':  str(cell_prop.area), 'Perimeter':  str(cell_prop.perimeter), 'Eccentricity':  str(cell_prop.eccentricity),\
                                 'Compactness':  str(cell_prop.compactness)})
            
                
def detect_not_rounded_cells(cell_props, th_compactness = 0.46):
    # print('Detecting not rounded cells')
    vector_compactness_per_cell = []
    vector_idx = []
    for cell_prop in cell_props:
        compactness = cell_prop.compactness
        vector_compactness_per_cell.append(compactness)
        
        vector_idx.append(cell_prop.label)
    
    not_compact = np.asarray(vector_compactness_per_cell) < th_compactness
    not_compact = np.nonzero(not_compact)
    not_compact_idx = np.take(vector_idx,not_compact)
    not_compact_idx = not_compact_idx[0]
    
    return vector_compactness_per_cell, not_compact_idx

def get_img_from_idx_cells(img_segmentation, list_idx):
    img_result = np.zeros_like(img_segmentation, dtype=img_segmentation.dtype)
    
    for i_cell in list_idx:
        img_result = np.where(img_segmentation == i_cell,img_segmentation,img_result)
        
    return img_result

def matching_label_pairs(matrix1, matrix2, min_pixels = 0):
    # Convert matrices to NumPy arrays
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)

    unique_labels_matrix1 = np.unique(array1)
    
    # Initialize a list to store matching label pairs
    matching_pairs = []

    # Iterate over unique labels in matrix1
    for label1 in unique_labels_matrix1:
        # Find indices where the label appears in matrix1
        indices_matrix1 = np.where(array1 == label1)

        # Extract corresponding labels from matrix2
        corresponding_labels_matrix2 = array2[indices_matrix1]

        # Iterate over unique labels in matrix2 corresponding to label1 in matrix1
        for label2 in np.unique(corresponding_labels_matrix2):
            
            n_pixels= np.count_nonzero(corresponding_labels_matrix2 == label2)
            if n_pixels>min_pixels:
                matching_pairs.append((label1, label2))
            
    matching_pairs_non_zero_left = [elem for elem in matching_pairs if elem[0] != 0]
    matching_pairs_non_zero = [elem for elem in matching_pairs if (elem[0] != 0 and elem[1] != 0)]

    return matching_pairs, matching_pairs_non_zero_left, matching_pairs_non_zero

def get_correspondance_segmentations(img_segmentation_a, img_segmentation_b, min_pixels_matching=0):
        
    #Correspondance nuclei to cell
    matching_pairs_a_to_b, matching_pairs_a_to_b_non_zero_left, matching_pairs_a_to_b_non_zero =\
        matching_label_pairs(img_segmentation_a, img_segmentation_b, min_pixels=min_pixels_matching)
    
    a_labels = np.array(matching_pairs_a_to_b_non_zero)[:, 0]
    unique_values_a_to_b, indices, counts = np.unique(a_labels, return_index=True, return_counts=True)
    #Repeated a labels
    repeated_indices = indices[counts > 1]
    a_labels_repeated = a_labels[repeated_indices]
    pairs_a_to_b_non_zero_repeated_binary = np.isin(a_labels, a_labels_repeated).astype(int)
    matching_pairs_a_to_b_non_zero_repeated = [matching_pairs_a_to_b_non_zero[i] for i in np.nonzero(pairs_a_to_b_non_zero_repeated_binary>0)[0]]
    
    #print(matching_pairs_nuclei_to_membrane_non_zero_repeated)
    #Plot
    
    # Left only nuclei and cells listed as with multiple nuclei
    mask = np.isin(img_segmentation_a, a_labels_repeated)
    img_segmentation_a_modified = np.where(mask, img_segmentation_a, 0)
    
    b_labels = np.array(matching_pairs_a_to_b_non_zero_repeated)[:, 1]
    mask = np.isin(img_segmentation_b, b_labels)
    img_segmentation_b_modified = np.where(mask, img_segmentation_b, 0)
    
    return img_segmentation_a_modified, img_segmentation_b_modified, a_labels_repeated

def get_join_properties(matching_pairs_a_to_b, props_a, props_b, suffixes=['_x', '_y']):
    
    df_a = pandas.DataFrame.from_records([a1.to_dict() for a1 in props_a])
    df_b = pandas.DataFrame.from_records([b1.to_dict() for b1 in props_b])
    df_a_b = pandas.DataFrame(matching_pairs_a_to_b,columns=["a", "b"])
    
    #resultdf=df_a.merge(df_b,how="inner",on="label")
    
    df_merge_a = df_a_b.merge(df_a, how='inner', on=None, left_on='a', right_on='label', suffixes=[suffixes[0],'_aux'])
    
    df_merge_n_n = df_merge_a.merge(df_b, how='inner', on=None, left_on='b', right_on='label', suffixes=suffixes)
    
    return df_merge_n_n


def get_cells_in_edges(cell_props_membrane, n_row, n_col):
    vector_labels_in_edge = []
    
    for cell_prop in cell_props_membrane:
        (min_row, min_col, max_row, max_col) = cell_prop.bbox
        
        if min_row <=0 or min_col <= 0 or max_row >= n_row or max_col >= n_col:
            vector_labels_in_edge.append(cell_prop.label)
    
    return vector_labels_in_edge

def delete_cells_in_edges(cell_props_membrane, vector_labels_in_edge):
    cell_props_membrane_not_in_edge = []
    
    for cell_prop in cell_props_membrane:
        if cell_prop.label in vector_labels_in_edge:
            cell_props_membrane_not_in_edge.append(cell_prop)
    
    return cell_props_membrane_not_in_edge

def delete_nuclei_of_cells_in_edges(matching_nuclei_membrane):
    #print(matching_nuclei_membrane)
    vector_pos_nuclei_to_delete = []
    vector_pos_nuclei_to_keep = []
    for pair in matching_nuclei_membrane:
        if pair[1]==0: #It means the nuclei matches background of the image
            vector_pos_nuclei_to_delete.append(pair[0])
        else:
            vector_pos_nuclei_to_keep.append(pair[0])
    return vector_pos_nuclei_to_delete, vector_pos_nuclei_to_keep

def get_density_bins(cell_props, img_width, img_height, axis=1, n_bins=20):
    vector_axis_pos = []
    
    for cell_prop in cell_props:
        if axis==0:
            vector_axis_pos.append(cell_prop.xCentroid)
        else:
            vector_axis_pos.append(cell_prop.yCentroid)
    #vector_axis_pos = np.unique(vector_axis_pos)
            
    count, edges = np.histogram(vector_axis_pos, bins=n_bins, range=(0, img_height))
    return count, edges

def get_centroids(cell_props, bbox = None):
    list_centroids = []
    for cell_prop in cell_props:
        if bbox is None:
            list_centroids.append([cell_prop.xCentroid,cell_prop.yCentroid])
        else:
            xmin, ymin, xmax, ymax = bbox
            if xmin <= cell_prop.xCentroid < xmax and ymin <= cell_prop.yCentroid < ymax:
                list_centroids.append([cell_prop.xCentroid,cell_prop.yCentroid])
    return list_centroids

def get_joint_expr_per_cell(img_segmentation, img_expression1, img_expression2, img_segmentation_channel1, img_segmentation_channel2):
    
    regions = measure.regionprops(img_segmentation)
    list_cell_expr = []
    for region in regions:
        
        binary_image = np.where(img_segmentation == region.label,1,0)
        
        masked_expression1 = img_expression1[binary_image > 0]
        masked_expression2 = img_expression2[binary_image > 0]
        masked_pred1 = img_segmentation_channel1[binary_image > 0]
        masked_pred2 = img_segmentation_channel2[binary_image > 0]
        
        mean_exp1 = np.median(masked_expression1) #mean can be used too
        mean_exp2 = np.median(masked_expression2)
        pred1 = np.int8(np.median(masked_pred1>0)) #is more classified as class 1?
        pred2 = np.int8(np.median(masked_pred2>0)) #is more classified as class 2?
        
        #label, expr1, expr2, pred1, pred2
        joint_expr = [region.label, mean_exp1, mean_exp2, pred1, pred2]
        list_cell_expr.append(joint_expr)

    return list_cell_expr

def filter_joint_cell_expr(list_cell_expr1_expr2): #Filtering if it is at least positive for one
    filtered_list_cell_expr1_expr2 = []
    for cell_expr1_expr2 in list_cell_expr1_expr2:
        #The prediction is in the last two columns.
        if cell_expr1_expr2[-1] + cell_expr1_expr2[-2] >0:
            filtered_list_cell_expr1_expr2.append(cell_expr1_expr2)
            
    return filtered_list_cell_expr1_expr2

def plot_expressions(list_cell_expr1_expr2, title_plot, label_x = 'Channel 1', label_y = 'Channel 2'):
    max_expr = 0;
    plt.figure(figsize=(8, 8))
    
    for cell_expr1_expr2 in list_cell_expr1_expr2:
        label, mean_exp1, mean_exp2, pred1, pred2 = cell_expr1_expr2
        if max_expr< np.max([mean_exp1,mean_exp2]):
            max_expr = np.max([mean_exp1,mean_exp2])
            
        plt.scatter(mean_exp1, mean_exp2, color='black', marker='o')
            
    plt.title(title_plot)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.ylim(0, max_expr + 0.01)
    plt.xlim(0, max_expr + 0.01)
    
    #plt.show()

def save_csv(list_cell_expr1_expr2, csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        csv_writer.writerow(["Label", "Median_Exp1", "Median_Exp2", "Pred1", "Pred2"])
        
        # Write each row of data
        for cell_expr1_expr2 in list_cell_expr1_expr2:
            csv_writer.writerow(cell_expr1_expr2)

def main():
    
    file_images = []
    for file in os.listdir(folder_original):
        if file.endswith(".png") or file.endswith(".bmp")  or file.endswith(".tif")  or file.endswith(".jpg"):
            file_images.append(file)
    
    it_debug = 0
    for file_image in file_images:
        if it_debug < 1:
            path_original = os.path.join(folder_original, file_image)
            path_segmentation = os.path.join(folder_segmentations, file_image + ending_segmentation)
            
            img_original = imread(path_original)
            img_segmentation = imread(path_segmentation)
            
            cell_props = get_props_per_cell(img_segmentation)
            
            #Get not compact cells
            vector_compactness_per_cell, not_compact_idx = detect_not_rounded_cells(cell_props)
            img_not_rounded_cells = get_img_from_idx_cells(img_segmentation, not_compact_idx)
            
            #Get big cells
            vector_area_per_cell, big_cells_idx = detect_big_cells(cell_props, th_size = th_size)
            img_big_cells = get_img_from_idx_cells(img_segmentation, big_cells_idx)
            
            
            if flag_show:
                #fig = plt.figure()
                
                plt.subplot(2, 4, 1)
                plt.imshow(img_original,cmap='gray')
                plt.gca().set_title('Original')
                plt.subplot(2, 4, 2)
                plt.imshow(img_segmentation)
                plt.gca().set_title('Segmentation - Total cells:' + str(len(cell_props)))
                
                plt.subplot(2, 4, 3)
                plt.hist(vector_compactness_per_cell, bins=20)
                plt.gca().set_title('Hist compactness')
                plt.subplot(2, 4, 4)
                plt.imshow(img_not_rounded_cells)
                plt.gca().set_title('Not compact cells: ' + str(len(not_compact_idx)))
                
                plt.subplot(2, 4, 5)
                plt.hist(vector_area_per_cell, bins=20)
                plt.gca().set_title('Hist area')
                plt.subplot(2, 4, 6)
                plt.imshow(img_big_cells)
                plt.gca().set_title('Big cells: ' + str(len(big_cells_idx)))
                
                
                figManager = plt.get_current_fig_manager()
                
                figManager.window.showMaximized()
                
                plt.show()
                
            
        it_debug = it_debug + 1
        
        
        
if __name__ == "__main__":
    main()
