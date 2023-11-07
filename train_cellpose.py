#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:33:19 2023

@author: lucas
"""
import os
import shutil
from aux_functions.CC_Cells import function_encode_annotations
from aux_functions.grid_search_cellpose import gridsearch

###################################   PARAMETERS   #########################


folder_rgb_images_training      = ''# Training images
folder_rgb_annotations_training = ''# Training images annotated

folder_rgb_images_validation      = ''# Validation images
folder_rgb_annotations_validation = ''# Validation images

vector_diameters = [5, 10, 15, 20, 25]
vector_models = ['nuclei', 'cyto', 'cyto2']


##############################################################################

def copy_images_to_folder(folder_orig, folder_dest):
    for file in os.listdir(folder_orig):
        if file.endswith(".png") or file.endswith(".bmp")  or file.endswith(".tif"):
            path_orig = os.path.join(folder_orig,file)
            path_dest = os.path.join(folder_dest,file)
            shutil.copy2(path_orig, path_dest)
            

def main():
    
    # Enconde training annotations and copy original images to the same folder for training
    folder_masks_encoded_training = folder_rgb_annotations_training + '_masks_python'
    function_encode_annotations(folder_rgb_annotations_training, folder_masks_encoded_training)
    copy_images_to_folder(folder_rgb_images_training, folder_masks_encoded_training)
    
    # Enconde validation annotations and copy original images to the same folder for validation
    folder_masks_encoded_validation = folder_rgb_annotations_validation + '_masks_python'
    function_encode_annotations(folder_rgb_annotations_validation, folder_masks_encoded_validation)
    copy_images_to_folder(folder_rgb_images_validation, folder_masks_encoded_validation)
    
    #Grid search to find the best architecture retrained
    best_model_path, best_agg_jaccard_index, best_diameter, best_pretrained_model = gridsearch(folder_masks_encoded_training, folder_masks_encoded_validation,\
                                                                                               vector_diameters, vector_models)
    print("----Best configuration---")
    print('JI: ' + str(best_agg_jaccard_index))
    print('Diameter: ' + str(best_diameter))
    print('Pretrained model: ' + str(best_pretrained_model))
    print('Model trained: ' + best_model_path)

if __name__ == "__main__":
    main()
