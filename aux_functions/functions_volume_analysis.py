#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:12:05 2025

@author: lucas
"""
from skimage.measure import regionprops
import numpy as np
from skimage.measure import marching_cubes
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def compute_surface_area(verts, faces):
    total_area = 0.0
    for face in faces:
        A, B, C = verts[face]  # Get triangle vertices
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))  # Triangle area # Cross product of the vector divided 2
        total_area += area
    return total_area

# Wadell in 1935
def get_sphericity(volume, surface_area):
    sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / (surface_area + 0.00000001)
    return sphericity
    
def compute_principal_length(binary_volume):
    """Compute the longest diagonal of an object using PCA."""
    coords = np.argwhere(binary_volume)  # Get object voxel coordinates
    if coords.shape[0] < 2:
        return 0  # Avoid PCA errors for single-voxel objects
    
    pca = PCA(n_components=3) # It is 3D
    pca.fit(coords)
    
    return 2 * np.sqrt(np.max(pca.explained_variance_))  # Length along the main axis


def get_objects_properties(labeled_array):
    props = regionprops(labeled_array)  # Compute properties
    data = []
    
    for region in props:

        binary_volume = labeled_array == region.label
        num_voxels = np.count_nonzero(binary_volume)
        verts, faces, _, _ = marching_cubes(binary_volume, 0.0)
        surface_area = compute_surface_area(verts, faces)
        length = compute_principal_length(binary_volume)
        sphericity = get_sphericity(num_voxels, surface_area)
        
        data.append([region.label, num_voxels, round(sphericity,3), round(length,3)])
        
        # print(f"Object ID {region.label}: Volume={num_voxels}, Sphericity={sphericity}")
    
    df = pd.DataFrame(data, columns=["Label", "Volume", "Sphericity", "Length"])
    return df

def plot_save_sphericity_histogram(df, filename):
    
    plt.figure(figsize=(8, 5))
    plt.hist(df["Sphericity"], bins=5, range=(0, 1), edgecolor="black", alpha=0.7)
    
    plt.xlabel("Sphericity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sphericity")
    plt.xlim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save the figure as PNG
    plt.close()  # Close the figure to free memory
    
    
def plot_save_length_histogram(df, filename):
    
    plt.figure(figsize=(8, 5))
    plt.hist(df["Length"], bins=5, edgecolor="black", alpha=0.7)

    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Length")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as PNG
    plt.close()  # Close the figure to free memory