# Segmentation and quantification of pericytes

Lucas D. Lo Vercio (lucasdaniel.lovercio@ucalgary.ca)

Alberta Children's Hospital Research Institute, University of Calgary (Calgary, AB, Canada)

## Introduction

## Main scripts

### segment_pericytes_folder.py

Script for segmentation and quantification of the body of pericytes in young zebrafish embryos, from tiff files in a folder.

## Materials

### Trained architectures

https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Cellpose_pretrained_models/trained_architectures_pericytes

## Input:

A folder containing .tif or .tiff files, where the channel of the vessel marker (odd numbers) and pericytes marker (even numbers) are intercalated.

Downloaded architecture from the OneDrive link above.

## Output

Endings:

_volume_pericytes.tiff: Volume containing only the pericyte marker channel.

_volume_vessel.tiff: Volume containing only the vessel marker channel.

_segmentation.tiff: Volume containing the segmentation of the pericytes' body. The value of the voxel represent the label of the pericyte it belongs to.

_segmentation_binary.tiff: Binarization of the segmentation volume, useful for visualization in Paraview.

_pericytes_properties.csv: Shape descriptors of each segmented pericyte.

_sphericity_histogram.png: A plot from the csv for preliminary assessment.

_length_histogram.png: A plot from the csv for preliminary assessment.

### Required libraries

cellpose
matplotlib
scikit-image
