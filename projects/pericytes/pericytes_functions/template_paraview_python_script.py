#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:53:19 2025

@author: lucas
"""

TEMPLATE_HEAD = '''from paraview.simple import *

from paraview import servermanager

def load_volume(tiff_path, registrationName, colormap_name):
    reader = TIFFSeriesReader(registrationName=registrationName, FileNames=[tiff_path])
    renderView1 = GetActiveViewOrCreate('RenderView')
    display = Show(reader, renderView1)
    display.UseSeparateColorMap = True
    
    # Set representation to Volume
    display.SetRepresentationType('Volume')
    # get color transfer function/color map for 'TiffScalars'
    tiffScalarsLUT = GetColorTransferFunction('TiffScalars')
    renderView1.ResetCamera(False, 0.9)
    

    # Color by the point data
    display.UseSeparateColorMap = 1
    ColorBy(display, ('POINTS', 'Tiff Scalars'), True)
    HideScalarBarIfNotNeeded(tiffScalarsLUT, renderView1)
    
    display.UseSeparateColorMap = True
    # Apply transfer functions
    display.RescaleTransferFunctionToDataRange(True, False)
    
    # show color bar/color legend
    display.SetScalarBarVisibility(renderView1, True)
    
    # get separate color transfer function/color map for 'TiffScalars'
    display_TiffScalarsLUT = GetColorTransferFunction('TiffScalars', display, separate=True)
    
    # get separate opacity transfer function/opacity map for 'TiffScalars'
    display_TiffScalarsPWF = GetOpacityTransferFunction('TiffScalars', display, separate=True)
    
    # get separate 2D transfer function for 'TiffScalars'
    display_TiffScalarsTF2D = GetTransferFunction2D('TiffScalars', display, separate=True)
    
    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    display_TiffScalarsLUT.ApplyPreset(colormap_name, True) 
    # Render()
    # Interact()
    
    return reader

# Load volumes
'''
