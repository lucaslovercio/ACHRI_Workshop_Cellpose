# Segmentation and quantification of cells and layers in IHC images of cerebral cortex

Lucas D. Lo Vercio (lucasdaniel.lovercio@ucalgary.ca)

Alberta Children's Hospital Research Institute, University of Calgary (Calgary, AB, Canada)

## Introduction

## Main scripts

### segment_projected_stack.py

Script for segmentation and quantification.

## Materials

### Trained architectures

https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Cellpose_pretrained_models/trained_architectures_cortex

### Required libraries

cellpose
matplotlib
scikit-image

## Output

### Segmentations

The ending "_segmentation.png" is a 16 bit image that represent the nuclei segmentation of each channel, while "_segmentation_masked.png" is the segmentation of the channel masked with the segmentation of the nuclei channel (ie, DAPI).

### Edges

A linear fitting of the cells in the edge is computed to analyze its rugosity. To remove outliers in the segmentation, the center of mass in a subimage is computed and the cells ± 2 standard deviations are preserved. The file ending in "_top_edge_std_errors.csv" show the standard error of the different fittings for a succession of subimage_width. The csv files "_fitting.csv" and the image "_edge_fitting.png" shows only the results of fitting for the value set in subimage_width.

### Distribution measures

The CSV file "_bins.csv" has the number of cells in each layer (after masking with nuclei) in each horizontal bin. The image "_histograms.png" is a of this CSV file.

Two distribution indexes are computed: The Clark-Evans index (Clark and Evans, Ecology, 1954; Petrere, Oecologia, 1985; Zheng et al., Nucleic Acids Research, 2022) and the Morisita index (Morisita, Res Popul Ecol, 1962; Amaral et al. Appl Ecol Env Res, 2014; Hayes and Castillo, ISPRS Int. J. Geo-Inf, 2017). The file ending in "_distribution_indexes.csv" shows the distribution index for each neural layer in  rectangle determined by the fitting described before. Each column is the distribution for the signal in the layer, for the same region in the nuclei channel (ie, DAPI), and the value of the index if the same number of cells is randomly distributed. As the Morisita index depend on the size of the sample unit/plot/squares, the file "_morisita_succession.png" shows its value through different sizes.
