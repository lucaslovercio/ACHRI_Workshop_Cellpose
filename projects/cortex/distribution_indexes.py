#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:16:14 2024

@author: lucas
"""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from quantify_segmentation import get_centroids
import pandas as pd
import random

def get_clark_evans_index(points, bbox):
    #Adapted from Zheng et al. 2022 Aquila a spatial omics database and analysis platform
    
    n = len(points)
    if n < 4:
        return -1

    xmin, ymin, xmax, ymax = bbox
    valid_points = []
    for point in points:
        x, y = point
        if xmin <= x < xmax and ymin <= y < ymax:
            valid_points.append(point)
    
    n = len(valid_points)
    
    tree = cKDTree(valid_points) #rapidly look up the nearest neighbors of any point

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    #Query the kd-tree for nearest neighbors
    #k = The list of k-th nearest neighbors to return
    #Why k=2? I want only one. But with k=1 does not work
    r = np.array([tree.query(p, k=2)[0][1] for p in valid_points]) #It is a vector of the r
    
    rho = n / area # rho in Clark and Evans 1954
    nnd_mean = np.mean(r) # mean of the series of distances to nearest neighbor.
    nnd_expected_mean = 1.0 / (2.0 * np.sqrt(rho)) #Table I, mean distance to nearest neighbor expected in an infinitely large...
    big_r = nnd_mean / nnd_expected_mean # Table I, row 8. The main metric.
    
    return big_r

def generate_random_points2(num_points, x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    points = np.random.rand(num_points, 2)
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
    return points.tolist()

def get_list_cuadrants(points, bbox, dx, dy):

    xmin, ymin, xmax, ymax = bbox
    dimX = int((xmax - xmin) / dx)
    dimY = int((ymax - ymin) / dy)
    quadrants = []
    
    # Iterate through each quadrant
    for i in range(dimX):
        for j in range(dimY):
            quadrant_points = []
            # Define the boundaries of the current quadrant
            x_min_q = xmin + i * dx
            x_max_q = xmin + (i + 1) * dx
            y_min_q = ymin + j * dy
            y_max_q = ymin + (j + 1) * dy
            

            # Check each point if it falls within the current quadrant
            for point in points:
                x, y = point
                if x_min_q <= x < x_max_q and y_min_q <= y < y_max_q:
                    quadrant_points.append(point)
            quadrants.append(quadrant_points)

    return quadrants

#Amaral 2014
def get_morisita_index(points, bbox, dx, dy):
    N = len(points)
    if N < 4:
        return -1
    
    xmin, ymin, xmax, ymax = bbox
    valid_points = []
    for point in points:
        x, y = point
        if xmin <= x < xmax and ymin <= y < ymax:
            valid_points.append(point)
    
    #N = len(valid_points)
    
    list_points_by_quadrant = get_list_cuadrants(valid_points, bbox, dx, dy)
    
    #Equation by Amaral et al.
    Q = len(list_points_by_quadrant)
    
    num1 = 0
    for l in list_points_by_quadrant:
        ni = len(l)
        num1 = num1 + ni * ni
    
    num2 = 0
    for l in list_points_by_quadrant:
        ni = len(l)
        num2 = num2 + ni
    
    den1 = num2 * num2
    den2 = num2
    I = Q * ( (num1 - num2) / (den1 - den2) )
    return I

def generate_random_points(bbox, n):
    xmin, ymin, xmax, ymax = bbox
    points = []
    for _ in range(n):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        points.append((x, y))
    return points


def get_morisita_index_succession(points, bbox, d_start = 50, d_step = 50, d_end = 250):
    
    windows = np.arange(d_start, d_end+1, d_step)
    list_morisita = []
    
    for d in windows:
    
        morisita_index_temp = get_morisita_index(points, bbox, d, d)
        list_morisita.append(morisita_index_temp)
        
    return list_morisita, windows
    

def generate_distribution_indexes(dims, cell_props_C1, cell_props_C2, cell_props_C3, cell_props_C4, C2_bbox, C3_bbox, C4_bbox, dx_morisita = 50, dy_morisita = 50):
            
    #The list is CellProperty, not a xy list! need to convert to xy list or modifify the scripts!!
    
    #Get centroids and plot dots in the bbox, both dapi and positive in the channel
    
    C2_centroids_bbox = get_centroids(cell_props_C2, C2_bbox)
    C2_centroids_bbox_x = [point[0] for point in C2_centroids_bbox]
    C2_centroids_bbox_y = [point[1] for point in C2_centroids_bbox]
    
    C3_centroids_bbox = get_centroids(cell_props_C3, C3_bbox)
    C3_centroids_bbox_x = [point[0] for point in C3_centroids_bbox]
    C3_centroids_bbox_y = [point[1] for point in C3_centroids_bbox]
    
    C4_centroids_bbox = get_centroids(cell_props_C4, C4_bbox)
    C4_centroids_bbox_x = [point[0] for point in C4_centroids_bbox]
    C4_centroids_bbox_y = [point[1] for point in C4_centroids_bbox]
    
    C1_centroids_C2_bbox =  get_centroids(cell_props_C1, C2_bbox)
    C1_centroids_C2_bbox_x = [point[0] for point in C1_centroids_C2_bbox]
    C1_centroids_C2_bbox_y = [point[1] for point in C1_centroids_C2_bbox]
    
    C1_centroids_C3_bbox =  get_centroids(cell_props_C1, C3_bbox)
    C1_centroids_C3_bbox_x = [point[0] for point in C1_centroids_C3_bbox]
    C1_centroids_C3_bbox_y = [point[1] for point in C1_centroids_C3_bbox]
    
    C1_centroids_C4_bbox =  get_centroids(cell_props_C1, C4_bbox)
    C1_centroids_C4_bbox_x = [point[0] for point in C1_centroids_C4_bbox]
    C1_centroids_C4_bbox_y = [point[1] for point in C1_centroids_C4_bbox]
    
    
    # print('C2')
    C2_morisita_index = get_morisita_index(get_centroids(cell_props_C2), C2_bbox, dx_morisita, dy_morisita)
    # print('C3')
    C3_morisita_index = get_morisita_index(get_centroids(cell_props_C3), C3_bbox, dx_morisita, dy_morisita)
    # print('C4')
    C4_morisita_index = get_morisita_index(get_centroids(cell_props_C4), C4_bbox, dx_morisita, dy_morisita)
    
    C2_clarkevans_index = get_clark_evans_index(get_centroids(cell_props_C2), C2_bbox)
    C3_clarkevans_index = get_clark_evans_index(get_centroids(cell_props_C3), C3_bbox)
    C4_clarkevans_index = get_clark_evans_index(get_centroids(cell_props_C4), C4_bbox)
    
    #What are the dispersion index for the DAPI in the same areas where the layers are segmented?
    
    C1_layer_C2_morisita_index = get_morisita_index(get_centroids(cell_props_C1), C2_bbox, dx_morisita, dy_morisita)
    C1_layer_C3_morisita_index = get_morisita_index(get_centroids(cell_props_C1), C3_bbox, dx_morisita, dy_morisita)
    C1_layer_C4_morisita_index = get_morisita_index(get_centroids(cell_props_C1), C4_bbox, dx_morisita, dy_morisita)
    
    C1_layer_C2_clarkevans_index = get_clark_evans_index(get_centroids(cell_props_C1), C2_bbox)
    C1_layer_C3_clarkevans_index = get_clark_evans_index(get_centroids(cell_props_C1), C3_bbox)
    C1_layer_C4_clarkevans_index = get_clark_evans_index(get_centroids(cell_props_C1), C4_bbox)
    
    
    # Compute the indexes for a random distribution of points in the bounding boxes
    C2_n = len(C2_centroids_bbox)
    C2_random = generate_random_points(C2_bbox, C2_n)
    C2_centroids_random_bbox_x = [point[0] for point in C2_random]
    C2_centroids_random_bbox_y = [point[1] for point in C2_random]
    C2_random_clarkevans_index = get_clark_evans_index(C2_random, C2_bbox)
    C2_random_morisita_index = get_morisita_index(C2_random, C2_bbox, dx_morisita, dy_morisita)
    
    C3_n = len(C3_centroids_bbox)
    C3_random = generate_random_points(C3_bbox, C3_n)
    C3_centroids_random_bbox_x = [point[0] for point in C3_random]
    C3_centroids_random_bbox_y = [point[1] for point in C3_random]
    C3_random_clarkevans_index = get_clark_evans_index(C3_random, C3_bbox)
    C3_random_morisita_index = get_morisita_index(C3_random, C3_bbox, dx_morisita, dy_morisita)
    
    C4_n = len(C4_centroids_bbox)
    C4_random = generate_random_points(C4_bbox, C4_n)
    C4_centroids_random_bbox_x = [point[0] for point in C4_random]
    C4_centroids_random_bbox_y = [point[1] for point in C4_random]
    C4_random_clarkevans_index = get_clark_evans_index(C4_random, C4_bbox)
    C4_random_morisita_index = get_morisita_index(C4_random, C4_bbox, dx_morisita, dy_morisita)

    
    distribution_indexes_table = {
            '--': ['C2','C3','C4'],
            'Clark_Evans': [C2_clarkevans_index,C3_clarkevans_index,C4_clarkevans_index ],
            'Clark_Evans_Nuclei': [C1_layer_C2_clarkevans_index,C1_layer_C3_clarkevans_index,C1_layer_C4_clarkevans_index],
            'Clark_Evans_Random': [C2_random_clarkevans_index,C3_random_clarkevans_index,C4_random_clarkevans_index],
            'Morisita': [C2_morisita_index,C3_morisita_index,C4_morisita_index],
            #'Morisita (debug)': [C2_morisita_index2,C3_morisita_index2,C4_morisita_index2],
            'Morisita_Nuclei': [C1_layer_C2_morisita_index,C1_layer_C3_morisita_index,C1_layer_C4_morisita_index],
            #'Morisita_Nuclei (debug)': [C1_layer_C2_morisita_index2,C1_layer_C3_morisita_index2,C1_layer_C4_morisita_index2],
            'Morisita_Random': [C2_random_morisita_index,C3_random_morisita_index,C4_random_morisita_index]
            #'Morisita_Random (debug)': [C2_random_morisita_index2,C3_random_morisita_index2,C4_random_morisita_index2]
    }

    # Create DataFrame
    #df_distribution_indexes = pd.DataFrame(distribution_indexes_table, index=['C2','C3','C4'])
    df_distribution_indexes = pd.DataFrame(distribution_indexes_table)
    
    plt.rcParams["figure.autolayout"] = True
    
    fig, ax = plt.subplots(2, 6, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
    
    plt.subplot(2, 6, 1)
    plt.scatter(C1_centroids_C2_bbox_x, C1_centroids_C2_bbox_y, s = 10)
    plt.title('DAPI in C2 layer')
    
    plt.subplot(2, 6, 2)
    plt.scatter(C2_centroids_bbox_x, C2_centroids_bbox_y, s = 10)
    plt.title('C2 Centroids')
    
    plt.subplot(2, 6, 3)
    plt.scatter(C2_centroids_random_bbox_x, C2_centroids_random_bbox_y, s = 10)
    plt.title('C2 Random')
    
    plt.subplot(2, 6, 4)
    plt.scatter(C1_centroids_C3_bbox_x, C1_centroids_C3_bbox_y, s = 10)
    plt.title('DAPI in C3 layer')
    
    plt.subplot(2, 6, 5)
    plt.scatter(C3_centroids_bbox_x, C3_centroids_bbox_y, s = 10)
    plt.title('C3 Centroids')
    
    plt.subplot(2, 6, 6)
    plt.scatter(C3_centroids_random_bbox_x, C3_centroids_random_bbox_y, s = 10)
    plt.title('C3 Random')
    
    plt.subplot(2, 6, 7)
    plt.scatter(C1_centroids_C4_bbox_x, C1_centroids_C4_bbox_y, s = 10)
    plt.title('DAPI in C4 layer')
    
    plt.subplot(2, 6, 8)
    plt.scatter(C4_centroids_bbox_x, C4_centroids_bbox_y, s = 10)
    
    plt.xlim(0, dims[1])  # Adjusting limits for better visualization
    plt.ylim(0, dims[0])
    plt.title('C4 Centroids')
    
    plt.subplot(2, 6, 9)
    plt.scatter(C4_centroids_random_bbox_x, C4_centroids_random_bbox_y, s = 10)
    plt.title('C4 Random')
    
    # Show plot
    plt.show()


    # Plot different morisita indexes    
    C2_list_morisita, C2_windows = get_morisita_index_succession(get_centroids(cell_props_C2), C2_bbox)
    C3_list_morisita, C3_windows = get_morisita_index_succession(get_centroids(cell_props_C3), C3_bbox)
    C4_list_morisita, C4_windows = get_morisita_index_succession(get_centroids(cell_props_C4), C4_bbox)
    
    # Random distribution to study
    C2_random_list_morisita, C2_windows = get_morisita_index_succession(C2_random, C2_bbox)
    C3_random_list_morisita, C3_windows = get_morisita_index_succession(C3_random, C3_bbox)
    C4_random_list_morisita, C4_windows = get_morisita_index_succession(C4_random, C4_bbox)
    
    morisita_handle = plt.figure()
    pltRandom, =plt.plot(C2_windows, np.ones_like(C2_windows),'-', color='black', linewidth=2.5, label='Random')
    
    plt.plot(C2_windows, C2_random_list_morisita,'--', color='blue')
    plt.plot(C3_windows, C3_random_list_morisita,'--', color='red')
    plt.plot(C4_windows, C4_random_list_morisita,'--', color='green')
    
    pltC2, = plt.plot(C2_windows, C2_list_morisita, color='blue', label='C2', linewidth=2.5)
    pltC3, = plt.plot(C3_windows, C3_list_morisita, color='red', label='C3', linewidth=2.5)
    pltC4, = plt.plot(C4_windows, C4_list_morisita, color='green', label='C4', linewidth=2.5)
    plt.legend(handles=[pltRandom, pltC2, pltC3, pltC4])
    plt.xlim(min(C2_windows), max(C2_windows) + 1)
    plt.ylim(0.75, 1.25)
    plt.xlabel('Quadrant size')
    plt.ylabel('Morisita index')
    plt.show()

    return df_distribution_indexes, morisita_handle
 
