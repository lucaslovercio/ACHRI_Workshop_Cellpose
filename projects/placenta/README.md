# Segmentation and Quantification of Nuclei and Cytoplasms

Lucas D. Lo Vercio (lucasdaniel.lovercio@ucalgary.ca)

Alberta Children's Hospital Research Institute, University of Calgary (Calgary, AB, Canada)

## Introduction

This project is to analyze morphology of nuclei and cytoplasms.

## Main scripts

### segment_quantify_nuclei_membrane.py

Script for segmentation and quantification of cells in the placenta, particularly to quantify the number of multinuclei cells. The Zo1 signal should be well defined, without cracks.

### script_compute_fusion_of_cells.py

Script for quantifying the fusion of cells, by segmenting the cells at different of continuity of the Zo1 signal. Along with the result of these segmentations, the nuclei segmentation and matching with cells complement the analysis.

### show_nuclei_membrane_segmentation.py

Script to display existing segmentations.

## Materials

### Trained architectures

https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Cellpose_pretrained_models

### Required libraries

cellpose
matplotlib
scikit-image

