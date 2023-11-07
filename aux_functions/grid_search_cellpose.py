#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:36:54 2023

@author: lucas
"""

from cellpose import io, models, metrics, plot
import shutil
import os
import numpy as np

def gridsearch(folder_training, folder_validation, vector_diameters, vector_models):
    
    best_agg_jaccard_index = 0
    best_model_path = ''
    best_diameter = 0
    best_pretrained_model = ''
    

    #Prepare inputs for model.train
    #From https://github.com/MouseLand/cellpose/blob/main/tests/test_train.py
    
    #Folder where to save the models
    #Not necessary to create, the folder models will be created in the training folder folder_training
    
    #Training data
    training_data = io.load_images_labels(folder_training, mask_filter='_masks')
    images_training, labels_training, image_names_training = training_data
    #Validation data
    validation_data = io.load_images_labels(folder_validation, mask_filter='_masks')
    images_validation, labels_validation, image_names_validation = validation_data
    channels = [0,0]
    
    flag_debug = True
    if flag_debug:
        print("len(images_training) " + str(len(images_training)))
        print("len(labels_training) " + str(len(labels_training)))
        print("len(image_names_training) " + str(len(image_names_training)))
        print("len(images_validation) " + str(len(images_validation)))
        print("len(labels_validation) " + str(len(labels_validation)))
        print("len(image_names_validation) " + str(len(image_names_validation)))
        
    for diameter in vector_diameters:
        for model_type in vector_models:
            
            print("Training - Diam: " + str(diameter) + ", model_type: " + model_type)
            
            #Load pretrained model
            model = models.CellposeModel(gpu=True, pretrained_model=False, model_type=model_type, diam_mean=diameter)
                        
            #Refine model with our data
            cpmodel_path = model.train(train_data = images_training, train_labels = labels_training, train_files=None,\
                                       test_data=images_validation, test_labels=labels_validation,\
                                           test_files=None, channels=channels, normalize=True, save_path=folder_training, save_every=100,\
                                               save_each=False, learning_rate=0.2, n_epochs=500, momentum=0.9, SGD=True,\
                                                   weight_decay=1e-05, batch_size=3, nimg_per_epoch=None, rescale=True,\
                                                       min_train_masks=5, model_name=None)
                
            masks_pred, flows, styles = model.eval(images_validation, diameter=diameter, channels=channels)
            
            agg_jaccard_index = metrics.aggregated_jaccard_index(labels_validation, masks_pred)
            
            if best_agg_jaccard_index < np.mean(agg_jaccard_index):
                best_agg_jaccard_index = np.mean(agg_jaccard_index)
                best_model_path = cpmodel_path
                best_diameter = diameter
                best_pretrained_model = model_type
                
                print('Temp Best JI: ' + str(best_agg_jaccard_index))
                print('Temp Best Diameter: ' + str(best_diameter))
                print('Temp Best Pretrained model: ' + str(best_pretrained_model))
                print('Temp Best Model trained: ' + best_model_path)
                print('---')
                

    return best_model_path, best_agg_jaccard_index, best_diameter, best_pretrained_model
