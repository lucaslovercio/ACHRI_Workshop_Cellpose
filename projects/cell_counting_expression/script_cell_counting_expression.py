
###################################   PARAMETERS   #########################

# Folder with the tiff images
folder_images = ''

folder_models = ''
# Just the name of the file with the model
model_nuclear       = '20251024_all_nuclei_20251022_model_cyto_diam_24_imgNormCellpose_imerodeFalse_ji_0.8379.574687'
model_nuclear_alive = '20251024_NonApoptotic_model_cyto2_diam_20_imgNormCellpose_imerodeFalse_ji_0.8225.392559'
model_nuclear_apoptotic = '20251024_apoptotic_20251022_model_nuclei_diam_12_imgNormCellpose_imerodeFalse_ji_0.7366.748566'
model_sox2          = '20251029_Sox2_model_nuclei_diam_27_ji_0.8448.447811'
model_tbr2          = '20250819_Tbr2_FromHistNorm_NormCellpose_model_cyto_diam_16_ji_0.8514.833700'
model_gpf           = '20250905_gfp_full_slide_model_nuclei_diam_40_ji_0.4079.101420'
model_gfap          = ''
model_tuj1          = '20250828_Tuj1_model_cyto2_diam_17_ji_0.5897.338790'

#List of corresponding channels. Recommended to start with the nuclei channel
list_channels           = [1,                     1,                   1,                    2,            3,           4]

# In the order of the channels you selected
list_names              = ['nuclear',         'alive',             'apoptotic',           'sox2',    'tbr2',        'Tuj1']
list_models             = [model_nuclear, model_nuclear_alive, model_nuclear_apoptotic, model_sox2, model_tbr2,   model_tuj1]

# To determine if 2 objects match or not
list_pixels_matching    = [36       ,         36,                   36,                     64,          64,          64]

# Usually these list and values are associated with the trained model (how it was trained)
list_norm               = [True,               True,                True,                  True,        True,          True]
list_gaussian_filter    = [False     ,        False,                False,                 False,        False,         False]

#Positions in list_names
segmentation_intensity  = [[1,2,3,4,5],[4,5,6]]

subfolder_output = '_analysis'
ending_edge = '_in_edge'
flag_gpu = False # In doubt, leave it as False
n_bins = 10 # For histograms

color_add = [100, 0, 0]

#########################################################################

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list
import os
from aux_functions.functionPercNorm import functionPercNorm
import numpy as np
from cellpose import models
from quantify_segmentation import matching_label_pairs, get_intensity_per_cell, get_intensity_around_cell
from quantify_segmentation import get_props_per_cell, get_labels
from aux_functions.draw_roi_over_image import draw_roi_over_image, overlap_mask_over_image_rgb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter

def power_of_two(n):
    return 2 ** n

def int_to_binary_array(n, width=None):
    binary_str = bin(n)[2:]  # Remove '0b' prefix
    if width is not None:
        binary_str = binary_str.zfill(width)  # Pad with zeros on the left
    return np.array([int(bit) for bit in binary_str])


def all_combinations(n_channels):
    n_all_combinations = power_of_two(n_channels)
    list_numbers = list(range(n_all_combinations))
    list_numbers = list_numbers[1:] # Remove 0
    list_combinations = []
    for number in list_numbers:
        binary_number = int_to_binary_array(number, n_channels)
        binary_number = binary_number[::-1] # Reversed
        list_combinations.append(binary_number)
    return list_combinations

def get_segmentation_and(list_segmentations, min_pixels=10):
    n_list = len(list_segmentations)
    if n_list<1:
        print("Wrong number of segmentations in list")
        sys.exit(1)
    img_seg_and = list_segmentations[0]
    for i in range(1, n_list):
        img_seg_2 = list_segmentations[i]
        _, _, matching_pairs_non_zero = matching_label_pairs(img_seg_and, img_seg_2, min_pixels=min_pixels)
        img_seg_and_temp = np.zeros_like(img_seg_and)
        for pair_nuclei_cell in matching_pairs_non_zero:
            img_seg_and_temp[img_seg_and == pair_nuclei_cell[0]] = pair_nuclei_cell[0]
        img_seg_and = img_seg_and_temp
        
    count_and = len(np.unique(img_seg_and)) - 1 # background does not count
    
    return img_seg_and, count_and

def plot_scatter_vs(df_seg_int, feature1_name, feature2_name, folder_output, file_image, seg_name):
    v_1 = df_seg_int[feature1_name]
    v_2 = df_seg_int[feature2_name]
    
    # Compute Pearson correlation coefficient
    if len(v_1)>2 and len(v_2)>2: #not empty
        r_pearson, p_value_pearson = pearsonr(v_1, v_2)
        rho_spearman, pval_spearman = spearmanr(v_1, v_2)
        # Linear fit (y = mx + b)
        m, b = np.polyfit(v_1, v_2, 1)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(v_1, v_2, s=50)#, alpha=0.7)
    if len(v_1)>2 and len(v_2)>2: #not empty
        plt.plot(v_1, b + m*v_1, 'r', label='fitted line')
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    #not empty
    if len(v_1)>2 and len(v_2)>2: #not empty
        plt.title(f'{feature1_name} vs {feature2_name} - Pearson r = {r_pearson:.3f}, Spearman rho = {rho_spearman:.3f} ')
    plt.savefig(os.path.join(folder_output, file_image + '_' + seg_name + '__' + feature1_name + '_vs_' + feature2_name + '.png'), dpi=400)
    
    plt.close('all')

def main():
    
    n_models = len(list_models)
    bins = np.linspace(0, 1, n_bins+1) # For intensity histograms
    
    
    file_images = []
    for file in os.listdir(folder_images):
        if file.endswith(".tiff") or file.endswith(".tif"):
            print(file)
            file_images.append(file)
            
    for file_image in file_images:
        print('---- Analyzing: ' + file_image + ' ----')
        
        path_image = os.path.join(folder_images, file_image)
        
        folder_output = os.path.join(folder_images, file_image + subfolder_output)
        if not os.path.exists(folder_output):
            os.makedirs(folder_output)
        
        images = read_multipage_tiff_as_list(path_image)
        n_channels_of_images = len(images)
        n_list_channels = len(list_channels)
        
        # Duplicating images in case different segmentations
        # Checking the input parameters
        if ( len(list_channels) != len(list_models) ) or ( min(list_channels) < 1 ) or ( max(list_channels) > n_channels_of_images ):
            print('Failed. Check the length of the list and the number of channels')
            sys.exit(1)
            
        images_temp = []
        images_norm_for_intensity = []
        #for channel_user in list_channels:
        for j in range(n_list_channels):
            channel_user = list_channels[j]
            image_temp = np.array(images[channel_user-1])
            if list_norm[j]:
                image_temp = functionPercNorm(image_temp)
            else:
                image_temp = np.double(np.double(image_temp) - np.min(image_temp))/np.double(np.max(image_temp) - np.min(image_temp))
            images_temp.append(image_temp) # Minus 1 as the human starts numbering in 1
            images_norm_for_intensity.append(functionPercNorm(np.array(images[channel_user-1])))
        
        images = images_temp
        
        n_channels = len(images) # Now the number of channels is different
        # Need to determine how many channels we will analyze
        effective_channels = np.min([n_channels, n_models])
        list_segmentations = []
        for channel_i in range(effective_channels):
            channel = images[channel_i]
            
            model_filename = list_models[channel_i]
            flag_gaussian_filter = list_gaussian_filter[channel_i]
            
            if flag_gaussian_filter:
                channel = gaussian_filter(channel, sigma=1)
            
            if model_filename != 'cyto3':
                model_fullpath = os.path.join(folder_models,model_filename)
                
                if not os.path.exists(folder_images):
                    print('Folder of images not found. Stopping execution.')
                    sys.exit(1)
                    
                if not os.path.exists(model_fullpath):
                    print(f"The folder '{model_fullpath}' does not exist.")
                    sys.exit(1)
                
                model = models.CellposeModel(pretrained_model=model_fullpath, gpu=flag_gpu)
                diameter = None
            else:
                model = models.CellposeModel(gpu=flag_gpu, model_type='cyto3')
                diameter = 50
            img_segmentation, _, _ = model.eval(channel, diameter = diameter, channels = [[0,0]], normalize=list_norm[channel_i])
            del model
            img_segmentation = np.uint16(img_segmentation)
            list_segmentations.append(img_segmentation)
        
        # Creating dataframe to save later the data
        list_columns = list_names.copy()
        list_columns.append('Count')
        df_matches = pd.DataFrame(columns=list_columns)
        
        # Generating segmentations and computing co-expression
        list_combinations = all_combinations(effective_channels)
        for combination in list_combinations:
            list_images_in_combination = []
            temp_list_pixels_matching = []
            for j_elem_comb in range(combination.shape[0]):
                elem_comb = combination[j_elem_comb]
                if elem_comb:
                    list_images_in_combination.append(list_segmentations[j_elem_comb])
                    temp_list_pixels_matching.append(list_pixels_matching[j_elem_comb])
            min_pixels_matching = np.max(np.array(temp_list_pixels_matching))
            segmentation_and, count_and = get_segmentation_and(list_images_in_combination, min_pixels=min_pixels_matching) 
            
            # Producing ROIs
            if np.sum(combination) > 1: #It is the AND of multiple channels, ROI in first channel (the last in the inversed list)
                img_and_rgb_detection = draw_roi_over_image(images[0], segmentation_and)
                image_uint8 = (images[0] * 255).astype(np.uint8)
                img_and_rgb_segmentation = overlap_mask_over_image_rgb(cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR), segmentation_and>0, color_add = color_add)
                
            else:
                channel_to_draw = np.where(combination)
                img_and_rgb_detection = draw_roi_over_image(images[channel_to_draw[0][0]], segmentation_and)
                image_uint8 = (images[channel_to_draw[0][0]] * 255).astype(np.uint8)
                img_and_rgb_segmentation = overlap_mask_over_image_rgb(cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR), segmentation_and>0, color_add = color_add)
            
            filename_img_and_det_rgb = os.path.join(folder_output, file_image + '_' + str(combination) + '_Count_' + str(count_and) + '_PxMatch' + str(min_pixels_matching) + '_detection.png')
            img_and_bgr_det = cv2.cvtColor(img_and_rgb_detection, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_img_and_det_rgb, img_and_bgr_det)
            
            filename_img_and_seg_rgb = os.path.join(folder_output, file_image + '_' + str(combination) + '_Count_' + str(count_and) + '_PxMatch' + str(min_pixels_matching) + '_segmentation.png')
            img_and_bgr_seg = cv2.cvtColor(img_and_rgb_segmentation, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename_img_and_seg_rgb, img_and_bgr_seg)
            
            row_strings = [str(boolean) for boolean in combination]
            row_strings.append(str(count_and))
            
            df_matches.loc[len(df_matches)] = row_strings
        
        # Save dataframe
        csv_output = os.path.join(folder_output, file_image + '_counts.csv')
        df_matches.to_csv(csv_output, index=False)    
        
        segmentations = segmentation_intensity[0]
        intensities = segmentation_intensity[1]
            
        for seg_i in segmentations:
            segmentation = list_segmentations[seg_i-1]
            seg_name = list_names[seg_i-1]
            props_cells = get_props_per_cell(segmentation)
            cells_id = get_labels(props_cells)
            data = {}
            data[seg_name] = cells_id
            for intensity_i in intensities:
                image_norm = images_norm_for_intensity[intensity_i-1]
                intensity_name = list_names[intensity_i-1]
                
                _, vector_mean_intensity = get_intensity_per_cell(cells_id, segmentation, image_norm)
                _, vector_mean_intensity_around_cell =get_intensity_around_cell(cells_id, segmentation, image_norm)
                
                
                data[intensity_name] = vector_mean_intensity
                data[intensity_name + ending_edge] = vector_mean_intensity_around_cell
                
                # Do histogram of intensities
                # For positive in seg_name, histogram of intensities in intensity_name
                
                plt.figure(figsize=(8, 5))
                counts, bin_edges, _ = plt.hist(vector_mean_intensity, bins=bins, edgecolor="black", alpha=0.7)
                plt.xlabel(intensity_name)
                plt.ylabel("Frequency")
                plt.title("For positive in " + seg_name + " intensities for " + intensity_name)
                plt.grid(axis="y", linestyle="--", alpha=0.6) 
                hist_output = os.path.join(folder_output, file_image + '_' + seg_name + '__intensities_for_' + intensity_name + '_histogram.png')
                plt.savefig(hist_output, dpi=300, bbox_inches="tight")  # Save as PNG
                plt.close()  # Close the figure to free memory
                
                # Save histogram data
                # bins give the edges; counts are the frequencies per bin
                hist_data = pd.DataFrame({
                    "bin_left": bin_edges[:-1],
                    "bin_right": bin_edges[1:],
                    "count": counts.astype(int)
                })
                
                csv_output = os.path.join(folder_output, file_image + '_' + seg_name + '__intensities_for_' + intensity_name + '_histogram.csv')
                hist_data.to_csv(csv_output, index=False)
                
                
            df_seg_intensity = pd.DataFrame(dict(data))
            csv_output = os.path.join(folder_output, file_image + '_' + seg_name +'.csv')
            df_seg_intensity.to_csv(csv_output, index=False)
            
            if len(intensities) > 1: #We can plot an scatter plot
                
                feature1_name = list_names[intensities[0]-1]
                feature2_name = list_names[intensities[1]-1]
    
                plot_scatter_vs(df_seg_intensity, feature1_name, feature2_name, folder_output, file_image, seg_name)
                
                #In edge
                plot_scatter_vs(df_seg_intensity, feature1_name + ending_edge, feature2_name + ending_edge, folder_output, file_image, seg_name)
            
            if len(intensities) > 2: #We can plot an scatter plot
                
                feature1_name = list_names[intensities[0]-1]
                feature2_name = list_names[intensities[2]-1]
                plot_scatter_vs(df_seg_intensity, feature1_name, feature2_name, folder_output, file_image, seg_name)
                
                plot_scatter_vs(df_seg_intensity, feature1_name + ending_edge, feature2_name + ending_edge, folder_output, file_image, seg_name)
                
                feature1_name = list_names[intensities[1]-1]
                feature2_name = list_names[intensities[2]-1]
                plot_scatter_vs(df_seg_intensity, feature1_name, feature2_name, folder_output, file_image, seg_name)
                
                plot_scatter_vs(df_seg_intensity, feature1_name + ending_edge, feature2_name + ending_edge, folder_output, file_image, seg_name)
    
if __name__ == "__main__":
    main()    

    
    
