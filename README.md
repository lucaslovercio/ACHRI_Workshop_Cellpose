# ACHRI Workshop Cellpose

Lucas D. Lo Vercio (lucasdaniel.lovercio@ucalgary.ca)

Alberta Children's Hospital Research Institute, University of Calgary (Calgary, AB, Canada)

## Introduction

In this repository you can find the publicly available source code to use and/or re-train Cellpose models for cell segmentation.

## Main scripts

### segment_folder.py

It retrieves images from a folder, creates a new one, and saves the segmentations in the new folder.

### segment_quantify_image.py

For a single image, it performs the segmentation, saving the segmentation as a PNG image and a CSV file with the code of each object (cell), its centroid and some features (Area, Perimeter, Eccentricity, Compactness).

### train_cellpose.py

Given training and validation data, and candidate diameters and pretrained models, it finds the best model taking into account the Aggregated Jaccard Index.

## Materials

### Trained architectures

https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Cellpose_pretrained_models

### Required libraries

cellpose
matplotlib
scikit-image

