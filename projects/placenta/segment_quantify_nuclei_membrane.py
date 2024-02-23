#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:37:27 2023

@author: lucas
"""
import sys
import os
import cv2
import numpy as np
from cellpose import models
from cellpose.io import imread
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from projects.placenta.show_nuclei_membrane_segmentation import figure_segmentation_matching_results
from quantify_segmentation import get_props_per_cell,\
    matching_label_pairs, get_join_properties, get_correspondance_segmentations,\
        get_cells_in_edges, delete_cells_in_edges, delete_nuclei_of_cells_in_edges, list_cells#, detect_big_cells
#    detect_not_rounded_cells, get_img_from_idx_cells,\
    
from aux_functions.functionPercNorm import functionPercNorm, get_one_channel
#import csv

###################################   PARAMETERS   #########################
folder = ''
input_nuclei    = folder + 'dapi_5.jpg'
input_membrane  = folder + 'membrane_5.jpg'

path_model_nuclei_trained  = 'DAPI_diam50_cyto2.792364'

#Can be changed
flag_normalize = True
flag_gpu = True
channels = [[0,0]] #Same channels as training
th_size=150
min_pixels_matching = 1500

#Change if not desired naming
csv_output                      = 'descriptors'
segmentation_nuclei_output        = input_nuclei + '_seg.png'
segmentation_membrane_output    = input_membrane + '_seg.png'

##############################################################################


#def main():

#Segment membrane
model_trained_membrane = models.Cellpose(model_type='cyto2', gpu=flag_gpu)

#Load image (first channel)
img_membrane_original = imread(input_membrane)
if isinstance(img_membrane_original, type(None)):
    print("---------------------------------------")
    print("---- File of membrane not found!!!!----")
    print("---- See path below -------------------")
    print("---------------------------------------")
    sys.exit(1)


img_membrane_original = get_one_channel(img_membrane_original)

if flag_normalize:
    img_membrane_original = functionPercNorm( np.single(img_membrane_original))

#Segment cells in image
img_segmentation_membrane, flows, styles, diams = model_trained_membrane.eval(img_membrane_original, diameter=None, channels= channels)

#Get centroids and shape descriptors
cell_props_membrane = get_props_per_cell(img_segmentation_membrane)

#Get cells in the edges of the image
[n_row, n_col] = img_segmentation_membrane.shape
labels_membrane_edge = get_cells_in_edges(cell_props_membrane,n_row, n_col)
#Delete cells in the edges from the list
cell_props_membrane = delete_cells_in_edges(cell_props_membrane, labels_membrane_edge)
#Delete those labels in image
for i in labels_membrane_edge:
    img_segmentation_membrane[img_segmentation_membrane == i]=0
    
#RE-do cell props membrane
cell_props_membrane = get_props_per_cell(img_segmentation_membrane)

#Save membrane segmentation
cv2.imwrite(segmentation_membrane_output, img_segmentation_membrane)

#Segment nucleis

#Segment nuclei
if not os.path.exists(path_model_nuclei_trained):
    print("----------------------------------------------------------")
    print("---- File of nuclei segmentation model not found!!!!-----")
    print("---- See path below --------------------------------------")
    print("----------------------------------------------------------")
    print(path_model_nuclei_trained)
    print("----------------------------------------------------------")
    sys.exit(1)

model_trained_nuclei = models.CellposeModel(pretrained_model=path_model_nuclei_trained, gpu=flag_gpu)

#Load image (first channel)
img_nuclei_original = imread(input_nuclei)
if isinstance(img_nuclei_original, type(None)):
    print("--------------------------------------")
    print("---- File of nuclei not found!!!!-----")
    print("---- See path below ------------------")
    print("--------------------------------------")
    sys.exit(1)

img_nuclei_original = get_one_channel(img_nuclei_original)

if flag_normalize:
    img_nuclei_original = functionPercNorm( np.single(img_nuclei_original))

#Segment nuclei in image
img_segmentation_nuclei, flows, styles = model_trained_nuclei.eval(img_nuclei_original, diameter=None, channels= channels)

#Get centroids and shape descriptors
cell_props_nuclei = get_props_per_cell(img_segmentation_nuclei)

#Get nuclei in the edges of the image
[n_row, n_col] = img_segmentation_nuclei.shape
labels_nuclei_edge = get_cells_in_edges(cell_props_nuclei,n_row, n_col)
#Delete nuclei in the edges from the list
cell_props_nuclei = delete_cells_in_edges(cell_props_nuclei, labels_nuclei_edge)
#Delete those nuclei labels in image
for i in labels_nuclei_edge:
    img_segmentation_nuclei[img_segmentation_nuclei == i]=0

#Matching nuclei and cells
matching_nuclei_membrane, _, matching_pairs_a_to_b_non_zero = matching_label_pairs(img_segmentation_nuclei, img_segmentation_membrane, min_pixels=min_pixels_matching)

#Delete those nuclei where membrane is 0
vector_pos_nuclei_to_delete, vector_label_nuclei_to_keep = delete_nuclei_of_cells_in_edges(matching_nuclei_membrane)
for label_nuclei_to_delete in vector_pos_nuclei_to_delete:
    img_segmentation_nuclei[img_segmentation_nuclei==label_nuclei_to_delete]=0

#Re-do shape descriptors
cell_props_nuclei = get_props_per_cell(img_segmentation_nuclei)
    
#Re-do the matching between nuclei and membrane
_, _, matching_pairs_a_to_b_non_zero = matching_label_pairs(img_segmentation_nuclei, img_segmentation_membrane, min_pixels=min_pixels_matching)


#Cells sharing nuclei and viceversa
print("Same nucleus in multiple cells")
img_segmentation_nuclei_sharing_cells, img_segmentation_membranes_sharing_nuclei, labels_nucleus_sharing_cells =\
    get_correspondance_segmentations(img_segmentation_nuclei, img_segmentation_membrane, min_pixels_matching=min_pixels_matching)
print(labels_nucleus_sharing_cells)

print("Multi-nuclei cells")
img_segmentation_membrane_multiple_nuclei, img_segmentation_nuclei_in_same_cells, labels_multinuclei_cells =\
    get_correspondance_segmentations(img_segmentation_membrane, img_segmentation_nuclei, min_pixels_matching=min_pixels_matching)
print(labels_multinuclei_cells)

print("Number of nuclei")
print(len(cell_props_nuclei))
print("Number of cells")
print(len(cell_props_membrane))

#Save nuclei segmentation
cv2.imwrite(segmentation_nuclei_output, img_segmentation_nuclei)

#Save this information to txt file
csv_output_nuclei = input_nuclei + '_labels_shared.txt'

file = open(csv_output_nuclei, 'a')
file.write("Labels of nucleus in multiple cells\n")
file.write(str(labels_nucleus_sharing_cells))
file.write("\nLabels of Multi-nuclei cells\n")
file.write(str(labels_multinuclei_cells))   
file.close()    

#Save CSVs
csv_output_nuclei = input_nuclei + '_nuclei_' + csv_output + '.csv'
list_cells(cell_props_nuclei, csv_output_nuclei)
csv_output_membrane = input_membrane + '_membrane_' + csv_output + '.csv'
list_cells(cell_props_membrane, csv_output_membrane)

suffixes=['_nucleus', '_cell']
dataframe_a_b = get_join_properties(matching_pairs_a_to_b_non_zero, cell_props_nuclei, cell_props_membrane, suffixes=suffixes)
csv_output_nuclei = input_nuclei + suffixes[0] + suffixes[1] +'_matched_' + csv_output + '.csv'
dataframe_a_b.to_csv(csv_output_nuclei, index=False)

#Remove nucleus in multiple cells (that cannot happen)
#labels_nuclei_sharing_cells
dataframe_a_b_clean = dataframe_a_b[~dataframe_a_b['a'].isin(labels_nucleus_sharing_cells)]
csv_output_nuclei_clean = input_nuclei + suffixes[0] + suffixes[1] +'_matched_no-nucleus-in-multiple-cells_' + csv_output + '.csv'
dataframe_a_b_clean.to_csv(csv_output_nuclei_clean, index=False)


#Show results
figure_segmentation_matching_results(img_membrane_original, img_segmentation_membrane, img_nuclei_original,\
                                     img_segmentation_nuclei,img_segmentation_membrane_multiple_nuclei, img_segmentation_nuclei_in_same_cells,\
                                         img_segmentation_nuclei_sharing_cells, img_segmentation_membranes_sharing_nuclei)