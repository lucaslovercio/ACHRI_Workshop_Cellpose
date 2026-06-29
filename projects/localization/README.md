# Localization

## Localization in 2D

Download the trained Cellpose architectures from the folder trained_architectures_localization in OneDrive.

The Python script script_localization_2_channels.py expects a folder of multipage TIFF files as input.

## Localization in 3D

Download the trained Cellpose architectures from the folder localization_3d in OneDrive.

Convert your vsi images to OME tiff using the ImageJ macro export_oir_channels_to_ometiff.ijm

In the terminal, first, run script_localization_3d_1_generate_csv_from_ome_tif.py to generate the segmentations and .csv file. Then, run script_localization_3d_2_generate_plots.py to generate the plots.

# Trained architectures

https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Cellpose_pretrained_models
