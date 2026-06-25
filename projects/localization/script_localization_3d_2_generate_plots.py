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
from quantify_segmentation import plot_expressions_labelled
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions_localization import compare_expression_around_speckle,\
    apply_colour_map, draw_rings_and_cake
from aux_functions.functions_plots import plot_violin_pct

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
        
        # Normalization for percentual expression
        df['total_expression'] = (df['Expression_in_speckle']
                                  + df['Expression_around_speckle']
                                  + df['Expression_outside_speckle'])
        
        df['Expression_in_speckle_perc']        = (df['Expression_in_speckle'] / df['total_expression'])
        df['Expression_around_speckle_perc']    = (df['Expression_around_speckle'] / df['total_expression'])
        df['Expression_outside_speckle_perc']   = (df['Expression_outside_speckle'] / df['total_expression'])

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
            
            plot_violin_pct(df, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Speckle_size'
            compare_expression_around_speckle(df,treatment1,treatment2,column=column)
            txt_comparison = 'SPECKLE-LEVEL-sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            plot_violin_pct(df, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)
            
            txt_output += txt_comparison + '\n'

            column='Expression_in_speckle_perc'
            compare_expression_around_speckle(df,treatment1,treatment2,column=column)
            txt_comparison = 'SPECKLE-LEVEL-sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            plt.close('all')
            txt_output += txt_comparison + '\n'

            plot_violin_pct(df, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)

            column='Expression_around_speckle_perc'
            compare_expression_around_speckle(df,treatment1,treatment2,column=column)
            txt_comparison = 'SPECKLE-LEVEL-sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            plt.close('all')
            txt_output += txt_comparison + '\n'

            plot_violin_pct(df, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)
            
        
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
                'Ratio_in_out': 'mean',
                'total_expression': 'mean',
                'Expression_in_speckle_perc': 'mean',
                'Expression_around_speckle_perc': 'mean',
                'Expression_outside_speckle_perc': 'mean'
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
            
            plot_violin_pct(df_mean, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Speckle_size'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Expression_in_speckle_perc'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            plot_violin_pct(df_mean, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)
            
            txt_output += txt_comparison + '\n'
            
            column='Expression_around_speckle_perc'
            compare_expression_around_speckle(df_mean,treatment1,treatment2,column=column)
            txt_comparison = 'NUCLEI-LEVEL-mean_sizes_of_speckles_'+ column + '_' + treatment1 + '_' + treatment2
            png_output_hist = os.path.join(vsi_folder_plots, txt_comparison +'.png')
            plt.savefig(png_output_hist, dpi=800)
            
            plot_violin_pct(df_mean, treatment_col = 'Treatment', col = column, ylabel = column, treatments = list_classes_effective)
            png_output_violin = os.path.join(vsi_folder_plots, 'violin_'+txt_comparison+'.png')
            plt.savefig(png_output_violin, dpi=800)
            
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
                    'Expression_outside_speckle',
                    'Expression_in_speckle_perc',
                    'Expression_around_speckle_perc',
                    'Expression_outside_speckle_perc'
                ]
            ]
            .mean()
        )
        
        csv_output_expressions = os.path.join(vsi_folder_plots, 'TREATMENT-LEVEL-mean_expressions.csv')
        df_treat_marker.to_csv(csv_output_expressions, index=False)
        
        # Create colour map
        list_suffix = ['_absExpr','_percExpr']
        list_expr_cols = [[
            'Expression_in_speckle',
            'Expression_around_speckle',
            'Expression_outside_speckle'],
            ['Expression_in_speckle_perc',
            'Expression_around_speckle_perc',
            'Expression_outside_speckle_perc']
            ]
        cmap = plt.get_cmap('Reds')
        for expr_cols,suf in zip(list_expr_cols, list_suffix):

            # ── Global normalisation (original behaviour) ─────────────────────────
            df_global = apply_colour_map(df_treat_marker, expr_cols, cmap = cmap, normalize_per_row=False)
            df_global.to_csv(
                os.path.join(vsi_folder_plots, 'expressions_colour_map'+suf+'_global'+'.csv'), index=False)
            columns = [s + "_RGB" for s in expr_cols]
            draw_rings_and_cake(df_global, vsi_folder_plots, columns = columns, cmap = cmap, suffix=suf+'_global')

            # ── Per-row normalisation ─────────────────────────────────────────────
            # Each Treatment/Speckle_marker pair is normalised independently:
            # the minimum value in the row → 0, the maximum → 1.
            df_per_row = apply_colour_map(df_treat_marker, expr_cols, cmap = cmap, normalize_per_row=True)
            df_per_row.to_csv(
                os.path.join(vsi_folder_plots, 'expressions_colour_map'+suf+'_per_row'+'.csv'), index=False)
            draw_rings_and_cake(df_per_row, vsi_folder_plots, columns = columns, cmap = cmap, suffix=suf+'_per_row')
        
        txt_output += '----------------------------\n\n'
        
        with open(output_txt, "w") as f:
            f.write(txt_output)
        
        plt.close('all')
        
    finally:
        print("------FINISHED--------")
    
if __name__ == "__main__":
    main()
