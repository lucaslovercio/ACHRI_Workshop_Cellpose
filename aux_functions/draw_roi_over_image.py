#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:28:52 2024

@author: lucas
"""

import numpy as np
import cv2

def overlap_mask_over_image_rgb(color_image, mask, color_add = [0, 50, 0]):
    color_image = np.float32(color_image) #need to convert to float to avoid overflow
    color_image[mask == 1] =  color_image[mask == 1] + color_add
    color_image[color_image>=255] = 255 #control values
    color_image = np.uint8(color_image) #back to uint8
    return color_image

def draw_mask_over_image_rgb(color_image, mask, color_mask = [0, 255, 0]):
    color_image[mask == 1] =  color_mask
    return color_image

def draw_mask_over_image(img_original_normalized, mask, color_mask = [0, 255, 0]):
    image_uint8 = (img_original_normalized * 255).astype(np.uint8)
    del img_original_normalized
    # Create a color image from the greyscale image for visualization
    color_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    color_image[mask == 1] =  color_mask # BGR color for green
    return color_image
    

def draw_roi_over_image(img_original_normalized, img_segmentation):
    image_uint8 = (img_original_normalized * 255).astype(np.uint8)
    del img_original_normalized
    # Find unique object labels, excluding the background label 0
    object_labels = np.unique(img_segmentation)
    object_labels = object_labels[object_labels != 0]
    
    # Create a color image from the greyscale image for visualization
    color_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    
    for label in object_labels:
        # Create a mask for the current object
        mask = img_segmentation == label
        
        # Find the bounding box coordinates for the current object
        y_indices, x_indices = np.where(mask)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Draw a red rectangle (BGR color space) around the object
        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    return color_image