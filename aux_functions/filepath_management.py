#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:20:15 2024

@author: lucas
"""

import os
import glob

def get_sample_name(path):
    head_tail = os.path.split(path)
    filename = head_tail[1]
    #print(filename)
    
    if filename.endswith('.tiff'):
        sample_name = filename[:-5]
    else: #end in .tif
        sample_name = filename[:-4]
    return sample_name


def get_image_filenames(folder_path, substring_pattern):
    # Use glob to find all .png and .jpg files in the folder
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    
    # Combine the lists of filenames
    image_filenames = png_files + jpg_files
    
    # Filter files that contain '_c1' in their filenames
    filtered_files = [os.path.basename(f) for f in image_filenames if substring_pattern in os.path.basename(f)]
    
    
    return filtered_files