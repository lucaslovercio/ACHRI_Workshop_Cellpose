#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################   PARAMETERS   #############################

vsi_folder = '' # Write the folder path, without the last \ or / . In Windows, place an r before the ''.
csv_filename = '' # Write the folder path, without the last \ or / . In Windows, place an r before the ''.
signal_of_interest = 'PRPF6'

###########################################################################

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from quantify_segmentation import get_expr_from_labels, matching_label_pairs, plot_expressions_labelled, generate_props_dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from functions_localization import draw_concentric_circles, plot_overlapping_distributions, compare_expression_around_speckle, cake_plot
import cv2


def main():
    
    try:
        
        txt_output = ''
        output_txt = os.path.join(vsi_folder,'experiment_output.txt')
        
        vsi_folder_plots = os.path.join(vsi_folder,'plots')
        os.makedirs(vsi_folder_plots, exist_ok=True)

        print('-- Generating Plots --')
        
        csv_input = os.path.join(vsi_folder, csv_filename)                    
        # Load CSV
        df = pd.read_csv(csv_input)

        # Retrieve lists
        list_all_label_image = df['Treatment'].tolist()
        list_all_speckle_expression = df['Expression_in_speckle'].tolist()
        list_all_outside_speckle_expression = df['Expression_outside_speckle'].tolist()
        
        plot_expressions_labelled(list_all_speckle_expression, list_all_outside_speckle_expression, list_all_label_image, 'expression', \
                                  label_x = 'in_speckle', label_y = 'outside_speckle', figsize = 8, flag_show = False)
        png_output_expressions = os.path.join(vsi_folder_plots, 'SPECKLE-LEVEL-expressions_voxels_in_and_outside.png')
        plt.savefig(png_output_expressions, dpi=800)
        
        list_classes_effective = list(set(list_all_label_image))
        
        if len(list_classes_effective) >= 2:
            # Mean per speckle
            column='Expression_around_speckle'
            treatment1 = list_classes_effective[0]
            treatment2 = list_classes_effective[1]
            
            column='Expression_in_speckle'
            compare_expression_around_speckle(df,treatment1,treatment2,column=column)
            txt_comparison = 'SPECKLE-LEVEL-expressions_per_speckle_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Ratio_in_out'
            compare_expression_around_speckle(df,treatment1,treatment2,column=column)
            txt_comparison = 'SPECKLE-LEVEL-expressions_per_speckle_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Speckle_size'
            compare_expression_around_speckle(df,treatment1,treatment2,column=column)
            txt_comparison = 'SPECKLE-LEVEL-sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
        
        df_mean = (
            df
            .groupby(['Image', 'Nuclei_id'], as_index=False)
            .agg({
                'Treatment': 'first',
                'Speckle_marker': 'first',
                'Speckle_channel': 'first',
                signal_of_interest + '_channel': 'first',
                'Speckle_size': 'mean',
                'Expression_in_speckle': 'mean',
                'Expression_around_speckle': 'mean',
                'Expression_outside_speckle': 'mean',
                'Ratio_in_around': 'mean',
                'Ratio_in_out': 'mean'
            })
        )
        
        csv_output_expressions = os.path.join(vsi_folder_plots, 'NUCLEI-LEVEL-mean_expressions.csv')
        df_mean.to_csv(csv_output_expressions, index=False)
        if len(list_classes_effective) >= 2:
            column='Expression_around_speckle'
            treatment1 = list_classes_effective[0]
            treatment2 = list_classes_effective[1]
            
            
            column='Expression_outside_speckle'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_expressions_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            
            column='Expression_in_speckle'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_expressions_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Ratio_in_out'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_expressions_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Speckle_size'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            if len(list_classes_effective) > 2:
                
                column='Expression_outside_speckle'
                treatment1 = list_classes_effective[0]
                treatment2 = list_classes_effective[2]
                compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
                txt_comparison = 'NUCLEI-LEVEL-mean_expressions_'+ column + '_' + treatment1 + '_' + treatment2
                png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
                plt.savefig(png_output_hist, dpi=800)
                
                txt_output += txt_comparison + '\n'
                
                column='Expression_in_speckle'
                compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
                txt_comparison = 'NUCLEI-LEVEL-mean_expressions_'+ column + '_' + treatment1 + '_' + treatment2
                png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
                plt.savefig(png_output_hist, dpi=800)
                
                txt_output += txt_comparison + '\n'
                
                column='Ratio_in_out'
                compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
                txt_comparison = 'NUCLEI-LEVEL-mean_expressions_'+ column + '_' + treatment1 + '_' + treatment2
                png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
                plt.savefig(png_output_hist, dpi=800)
        
        
        # Mean per marker and treatment
        
        df_treat_marker = (
            df_mean
            .groupby(['Treatment', 'Speckle_marker'], as_index=False)[
                [
                    'Expression_in_speckle',
                    'Expression_around_speckle',
                    'Expression_outside_speckle'
                ]
            ]
            .mean()
        )
        
        csv_output_expressions = os.path.join(vsi_folder_plots, 'TREATMENT-LEVEL-mean_expressions.csv')
        df_treat_marker.to_csv(csv_output_expressions, index=False)
        
        # Create colour map
        expr_cols = [
            'Expression_in_speckle',
            'Expression_around_speckle',
            'Expression_outside_speckle'
        ]
        
        vmin = df_treat_marker[expr_cols].min().min()
        vmax = df_treat_marker[expr_cols].max().max()
        
        cmap = cm.get_cmap('Reds')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        for col in expr_cols:
            df_treat_marker[col + '_RGB'] = df_treat_marker[col].apply(
                lambda x: cmap(norm(x))[:3]
            )
        
        csv_output_expressions = os.path.join(vsi_folder_plots, 'expressions_marker_treatment_colour_map.csv')
        df_treat_marker.to_csv(csv_output_expressions, index=False)
        
        # Draw concentric rings
        list_rings = []
        list_treatments = []
        list_markers = []
        for _, row in df_treat_marker.iterrows():
            expr_in = row['Expression_in_speckle_RGB']
            expr_around = row['Expression_around_speckle_RGB']
            expr_out = row['Expression_outside_speckle_RGB']
            rgb_list = [expr_out, expr_around, expr_in]
            img_concentric_circles = draw_concentric_circles(rgb_list,radius=300)
            treatment = row['Treatment']
            marker = row['Speckle_marker']
            list_treatments.append(treatment+'_'+marker)
            list_markers.append(marker)
            img_filename = treatment + '_' + marker + '.png'
            cv2.imwrite(os.path.join(vsi_folder_plots, img_filename), img_concentric_circles)
            list_rings.append(img_concentric_circles)
        
        # Save Cake Chart
        print('-- Generating Cake Plot --')
        img_cake_pie = cake_plot(list_rings, list_treatments, cmap)
        img_cake_pie_filename = 'cake_plot.png'
        cv2.imwrite(os.path.join(vsi_folder_plots, img_cake_pie_filename), img_cake_pie)
        
        txt_output += '----------------------------\n\n'
        
        with open(output_txt, "w") as f:
            f.write(txt_output)
        
        plt.close('all')
        
    finally:
        print("------FINISHED--------")
    
if __name__ == "__main__":
    main()
