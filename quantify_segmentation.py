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

###################################   PARAMETERS   #########################

folder_original = ''
folder_segmentations = ''
ending_segmentation = '_cellpose.png'
flag_show = True
th_size = 10000

##############################################################################

class CellProperty:
  def __init__(self, xCentroid, yCentroid,label, area, perimeter, eccentricity):
    self.xCentroid = xCentroid
    self.yCentroid = yCentroid
    self.label = label
    self.area = area
    self.perimeter = perimeter
    self.eccentricity = eccentricity

def get_props_per_cell(img_segmentation):
    # print('Getting properties of cells')
    # print(type(img_segmentation))
    # print(img_segmentation.shape)
    
    regions = measure.regionprops(img_segmentation)
    regionprops_selected = []
    for region in regions:
        
        binary_image = np.where(img_segmentation == region.label,1,0)
        perimeter = measure.perimeter(binary_image, neighborhood=8)
        #y is first dimension, x is second dimension
        cell_property = CellProperty(region.centroid[1], region.centroid[0], region.label, region.area, perimeter, region.eccentricity)
        regionprops_selected.append(cell_property)
    
    return regionprops_selected

def get_compactness(area, perimeter):
    compactness = (4. * np.pi * area.astype('float')) / (perimeter.astype('float') * perimeter.astype('float'))
    return compactness
    
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
    #big_cells_idx = vector_idx[np.nonzero(big_cells)] #O (background) is not in cellprops list
    big_cells_idx = big_cells_idx[0] #+ 1
    
    return vector_area_per_cell, big_cells_idx

def list_cells(cell_props, csv_output):
    with open(csv_output, 'w', newline='') as csvfile:
        fieldnames = ['Label', 'xCentroid', 'yCentroid', 'Area', 'Perimeter', 'Eccentricity', 'Compactness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        for cell_prop in cell_props:
            compactness = get_compactness(cell_prop.area, cell_prop.perimeter)
            writer.writerow({'Label': str(cell_prop.label), 'xCentroid': str(cell_prop.xCentroid), 'yCentroid': str(cell_prop.yCentroid),\
                             'Area':  str(cell_prop.area), 'Perimeter':  str(cell_prop.perimeter), 'Eccentricity':  str(cell_prop.eccentricity),\
                                 'Compactness':  str(compactness)})
            
                
def detect_not_rounded_cells(cell_props, th_compactness = 0.46):
    # print('Detecting not rounded cells')
    vector_compactness_per_cell = []
    vector_idx = []
    for cell_prop in cell_props:
        area = cell_prop.area
        perimeter = cell_prop.perimeter
        
        compactness = get_compactness(area, perimeter)
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