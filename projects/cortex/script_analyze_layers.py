#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################   PARAMETERS   #########################

folder_root = ''
tiffs_name = ['','']
tiffs_treatment = ['KO', 'WT']
channel_1_number, channel_1_name = 3, 'Tbr1'
channel_2_number, channel_2_name = 4, 'Bcl11b'
channel_3_number, channel_3_name = 2, 'EdU' # No data

# Folder with the trained architectures. It has the subfolders Models_for_layer_delineation (one
# model per layer channel, see LAYER_MODEL_FILES below: DAPI, SATB2, BCL11B, TBR1) and Models_EdU
# (EDU_MODEL_FILE). The model is chosen by the channel's name, so name the channels DAPI, SATB2, BCL11B,
# TBR1, EdU (any case).
folder_models = ''

# For expression (Fig 3)
flag_normalization_expression = True
output_folder_ending        = '_20261002'
output_folder_expression    = '_figure3_20261002'
output_folder_layers        = '_figure5_20261002'

# DAPI (nuclei reference) channel, numbered as in the image viewer starting at 1 - usually 1, but
# change it if the images were scanned in another order. Its name must be DAPI (that is how its model
# is found). Internally the layer channels are reordered (DAPI, SATB2, BCL11B, TBR1), but all
# labels/file names keep the original channel numbers (C1, C2, ...) as in the viewer.
channel_DAPI_number = 1
channel_DAPI_name = 'DAPI'


# Channels (number starting at 1, name, number of bins) to compute the per-bin mean/median expression
# for (Fig 3): [top_y, bottom_y] is divided into that many equal-height Y bins, independently
# per channel. Each channel is saved to its own csv per sample, then aggregated across samples.
# A channel number of None (e.g. no EdU staining in these tiffs) is skipped.
channels_binned_expression = [(channel_2_number, channel_2_name, 4), (channel_3_number, channel_3_name, 10)]

#Number of bins for histogram used for layers delinations (Fig 5)
n_bins = 20

# EdU (comment 8): number of bins of the histogram of the EdU positives, between the DAPI layer limits
n_bins_edu = 20

# Fig 3F: per-cell Bcl11b/Tbr1 intensity ratio (log2) histogram, one sample at a time for now
# (samples will later be combined into a single multi-age plot, like the reference figure).
ratio_bin_min = -5
ratio_bin_max = 2.5
ratio_n_bins = 10
min_cells_bin_dapi = 70
min_cells_bin_filter_layer_2 = 5
min_cells_bin_filter_layer_3_4 = 5

# Optional parameters for linear fitting of edge (Fig 5)
subimage_width = 100 # Width of the vertical strips where the extreme (top/bottom) cell is selected
flag_text = True #If show the error of the linear fitting
text_shift = 60 # How much above the linear fitting the text can go
fitting_color = 'green' # Color of line and text
fontsize = 5
marker_size = 10

#Parameters for running the segmentation (False: layer channels segmented without normalizing, as in segment_projected_stack.py)
flag_normalize = False
flag_gpu = True

# Dashed line (top_y/bottom_y markers) drawn on the rgb_with_lines debug image - run once,
# then check the dash spacing in the output image and adjust these two.
dash_length_px = 10
gap_length_px = 6

############################################################################

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
import math
from dataclasses import dataclass
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from cellpose import models

from quantify_segmentation import get_props_per_cell, get_density_bins
from aux_functions.functionPercNorm import functionPercNorm, functionLinearNorm
from aux_functions.functionReadTIFFMultipage import read_multipage_tiff_as_list, split_list_images, get_projected_image
from analyze_neuron_layers import plot_nuclei_segmentations_list, get_distribution_histograms_list, get_layer_nuclei_histogram,\
        get_segmentation_filtered_layer, get_number_of_cells, plot_binned_expression_boxplots, plot_generate_joint_expression_data,\
        get_top_cells_labels, fit_cells, plot_cells, get_different_fittings_histogram_list


# ── Layer segmentation models ─────────────────────────────────────────────────

# Each layer channel is segmented with its own Cellpose model (the 4 architectures of the article, as in
# segment_projected_stack.py), chosen by the channel's name (any case): {name: model file in
# <folder_models>/Models_for_layer_delineation/}. The order matters: DAPI first, then the layers from
# top (SATB2, upper layers) to bottom (BCL11B, TBR1, deep layers) - the layer channels are processed in
# this order, which the layer delineation (analyze_layers) relies on. Channels with any other name
# (e.g. EdU, an mRNA) have no layer segmentation: they can still be quantified in the DAPI cells (binned expression).
SUBFOLDER_LAYER_MODELS = 'Models_for_layer_delineation'
LAYER_MODEL_FILES = {
    'DAPI':   'Neurons_C1_DAPI.183326',
    'SATB2':  'Neurons_C2_SATB2.919883',
    'BCL11B': 'Neurons_C3_BCL11B.981474',
    'TBR1':   'Neurons_C4_TBR1.909737',
}

# EdU: a channel named EdU (any case) is segmented with its own model (Cellpose 2 re-trained), on the
# percentile-normalized channel. It is not a layer channel (no layer delineation). Its outputs go to the
# expression (Fig 3) folder, with the file names ending in ENDING_EDU_OUTPUTS (comment 8 of the reviewer).
EDU_CHANNEL_NAME = 'EDU'
SUBFOLDER_EDU_MODEL = 'Models_EdU'
EDU_MODEL_FILE = 'model_positives_EdU.253538'
ENDING_EDU_OUTPUTS = '_comment8'


def get_layer_key(name):
    """'DAPI', 'SATB2', 'BCL11B' or 'TBR1' if name is one of the layer channels (any case), else None."""
    if not name:
        return None
    key = name.strip().upper()
    return key if key in LAYER_MODEL_FILES else None


def get_layer_model_path(folder_models, layer_key):
    return os.path.join(folder_models, SUBFOLDER_LAYER_MODELS, LAYER_MODEL_FILES[layer_key])


def get_used_layer_keys(channel_names):
    """Layer keys of the named channels ({number: name}), in layer order."""
    keys = {get_layer_key(name) for name in channel_names.values()}
    return [key for key in LAYER_MODEL_FILES if key in keys]


def is_edu_channel(name):
    return bool(name) and name.strip().upper() == EDU_CHANNEL_NAME


def get_edu_model_path(folder_models):
    return os.path.join(folder_models, SUBFOLDER_EDU_MODEL, EDU_MODEL_FILE)


def get_edu_channel_numbers(channel_names):
    """Viewer numbers of the channels named EdU (any case) in {number: name} (at most one, see validate_config)."""
    return [number for number, name in channel_names.items() if is_edu_channel(name)]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class AnalyzeLayersConfig:
    """Everything a Figure 3 run needs. Channel numbers are as in the image viewer, starting at 1.
    The optional fields default to the values in the PARAMETERS section at the top of this script
    (that is also where the GUI, script_analyze_layers_gui.py, takes its defaults from)."""
    folder_root: str
    tiffs_name: list
    tiffs_treatment: list
    channel_names: dict                 # {viewer channel number: name}, e.g. {1: 'DAPI', 2: 'EdU', 3: 'Tbr1', 4: 'Bcl11b'}
    channel_1_number: int               # joint expression X axis (and denominator of the Fig 3F ratio); any named channel but DAPI
    channel_2_number: int               # joint expression Y axis (and numerator of the Fig 3F ratio); any named channel but DAPI
    channels_binned_expression: list    # [(viewer channel number, name, n_bins), ...]
    folder_models: str                  # folder with the Models_for_layer_delineation subfolder

    # Optional parameters (defaults = PARAMETERS section above)
    flag_normalization_expression: bool = flag_normalization_expression
    output_folder_ending: str = output_folder_ending
    output_folder_expression: str = output_folder_expression
    output_folder_layers: str = output_folder_layers
    n_bins: int = n_bins
    n_bins_edu: int = n_bins_edu
    ratio_bin_min: float = ratio_bin_min
    ratio_bin_max: float = ratio_bin_max
    ratio_n_bins: int = ratio_n_bins
    min_cells_bin_dapi: int = min_cells_bin_dapi
    min_cells_bin_filter_layer_2: int = min_cells_bin_filter_layer_2
    min_cells_bin_filter_layer_3_4: int = min_cells_bin_filter_layer_3_4
    marker_size: float = marker_size
    subimage_width: int = subimage_width
    flag_text: bool = flag_text
    text_shift: float = text_shift
    fitting_color: str = fitting_color
    fontsize: float = fontsize
    flag_normalize: bool = flag_normalize
    flag_gpu: bool = flag_gpu
    dash_length_px: int = dash_length_px
    gap_length_px: int = gap_length_px

    @property
    def channel_1_name(self):
        return self.channel_names[self.channel_1_number]

    @property
    def channel_2_name(self):
        return self.channel_names[self.channel_2_number]


# Optional parameters shown (and editable) in the GUI's second tab: (field name, description).
OPTIONAL_PARAMETERS = [
    ('flag_normalization_expression', 'Percentile-normalize channels before quantifying expression (Fig 3)'),
    ('output_folder_ending',          'Suffix of each sample output folder'),
    ('output_folder_expression',      'Suffix of the expression (Fig 3) outputs'),
    ('output_folder_layers',          'Suffix of the layers (Fig 5) output folder'),
    ('n_bins',                        'Number of bins of the histograms used for layer delineation (Fig 5)'),
    ('n_bins_edu',                    'Number of bins of the EdU positives histogram, between the DAPI layer limits (comment 8)'),
    ('ratio_bin_min',                 'Fig 3F ratio histogram: X axis minimum (log2)'),
    ('ratio_bin_max',                 'Fig 3F ratio histogram: X axis maximum (log2)'),
    ('ratio_n_bins',                  'Fig 3F ratio histogram: number of bins'),
    ('min_cells_bin_dapi',            'Layer delineation: min cells per bin, DAPI'),
    ('min_cells_bin_filter_layer_2',  'Layer delineation: min cells per bin, first channel after DAPI'),
    ('min_cells_bin_filter_layer_3_4','Layer delineation: min cells per bin, remaining channels'),
    ('marker_size',                   'Line width (/10) of the layer limits and fitted lines, and size of the extreme cells (Fig 5 figures)'),
    ('subimage_width',                'Edge fitting: width (px) of the vertical strips where the extreme cell is selected'),
    ('flag_text',                     'Edge fitting: write the std error of each fitted line'),
    ('text_shift',                    'Edge fitting: how much above the fitted line the std error text goes (px)'),
    ('fitting_color',                 'Edge fitting: color of the fitted lines and text (matplotlib color name)'),
    ('fontsize',                      'Edge fitting: font size of the std error text'),
    ('flag_normalize',                'Percentile-normalize the layer channels before their segmentation (off as in segment_projected_stack.py)'),
    ('flag_gpu',                      'Use the GPU for Cellpose'),
    ('dash_length_px',                'Dashed layer-limit lines: dash length (px)'),
    ('gap_length_px',                 'Dashed layer-limit lines: gap length (px)'),
]


def build_config_from_parameters():
    """AnalyzeLayersConfig from the PARAMETERS section at the top of this script (for running it without the GUI)."""
    channel_names = {channel_DAPI_number: channel_DAPI_name}
    for number, name in [(channel_1_number, channel_1_name), (channel_2_number, channel_2_name), (channel_3_number, channel_3_name)]:
        if number is not None:
            channel_names[number] = name
    return AnalyzeLayersConfig(folder_root=folder_root, tiffs_name=tiffs_name, tiffs_treatment=tiffs_treatment,
                         channel_names=channel_names,
                         channel_1_number=channel_1_number, channel_2_number=channel_2_number,
                         channels_binned_expression=channels_binned_expression,
                         folder_models=folder_models)


def validate_config(config):
    """Raises ValueError/FileNotFoundError with a readable message if the config can't be run."""
    if len(config.tiffs_treatment) != len(config.tiffs_name):
        raise ValueError(f'tiffs_treatment has {len(config.tiffs_treatment)} entries but tiffs_name has {len(config.tiffs_name)} - they must match.')

    keys = [get_layer_key(name) for name in config.channel_names.values() if get_layer_key(name)]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Each layer channel ({", ".join(LAYER_MODEL_FILES)}) can be used only once: {keys}')
    if 'DAPI' not in keys:
        raise ValueError('One channel must be named DAPI (the nuclei reference).')

    # Joint expression channels: any named channel but DAPI. A non-layer channel (e.g. EdU, an mRNA) has no
    # segmentation nor layer limits, see analyze_joint_expression.
    for number in (config.channel_1_number, config.channel_2_number):
        name = config.channel_names.get(number)
        if not name or get_layer_key(name) == 'DAPI':
            raise ValueError(f'The joint expression channels must be named channels other than DAPI; channel {number} is "{name}".')
    if config.channel_1_number == config.channel_2_number:
        raise ValueError('The two joint expression channels must be different.')

    edu_numbers = get_edu_channel_numbers(config.channel_names)
    if len(edu_numbers) > 1:
        raise ValueError(f'Only one channel can be named EdU: channels {edu_numbers}')

    if not config.folder_models:
        raise ValueError('Set folder_models (the folder with the ' + SUBFOLDER_LAYER_MODELS + ' and ' + SUBFOLDER_EDU_MODEL + ' subfolders).')
    paths_models = [get_layer_model_path(config.folder_models, key) for key in get_used_layer_keys(config.channel_names)]
    if edu_numbers:
        paths_models.append(get_edu_model_path(config.folder_models))
    missing = [path for path in paths_models if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError('Segmentation model(s) not found:\n' + '\n'.join(missing))


# ── Input helpers (also used by the GUI) ──────────────────────────────────────

def list_tiffs_in_folder(folder):
    """Sorted names of the .tif/.tiff files directly inside folder."""
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(folder, f)))


def get_number_of_pages(tiff_path):
    with Image.open(tiff_path) as img:
        return getattr(img, 'n_frames', 1)


def read_tiff_channels(tiff_path):
    """Each page of the multipage tiff is already a Z-max-projection, i.e. one channel.
    Returns the list of channels as 2D arrays, in the file's (viewer's) order."""
    images = read_multipage_tiff_as_list(tiff_path)
    n_channels = len(images)
    list_list_images = split_list_images(images, n_channels)
    return [get_projected_image(imgs) for imgs in list_list_images]


def get_channel_label(channel_number, channel_names):
    """'C<number>_<name>' (e.g. 'C3_Tbr1') if the channel has a name, else 'C<number>'."""
    name = channel_names.get(channel_number)
    return 'C' + str(channel_number) + ('_' + name if name else '')


def order_channels(list_numpydata, channel_names):
    """Internal channel order: first the layer channels of this image in layer order (DAPI, SATB2, BCL11B,
    TBR1 - the ones present), then the other channels (e.g. EdU, unnamed) in the file's order.
    Returns (reordered channels, channel_order, channel_labels, layer_keys), where channel_order[i] is the
    original 0-based index of internal channel i, channel_labels keeps the viewer's numbering + names and
    layer_keys are the keys of the first len(layer_keys) internal channels (the layer channels)."""
    n_channels = len(list_numpydata)
    key_to_index = {}
    for i in range(n_channels):
        key = get_layer_key(channel_names.get(i + 1))
        if key:
            key_to_index[key] = i
    layer_keys = [key for key in LAYER_MODEL_FILES if key in key_to_index]
    layer_indices = [key_to_index[key] for key in layer_keys]
    channel_order = layer_indices + [i for i in range(n_channels) if i not in layer_indices]
    list_reordered = [list_numpydata[i] for i in channel_order]
    channel_labels = [get_channel_label(i + 1, channel_names) for i in channel_order]
    return list_reordered, channel_order, channel_labels, layer_keys


def to_internal_index(channel_number, channel_order):
    """Viewer channel number (starting at 1) -> index into the reordered (DAPI first) channel lists.
    Numbers outside the image are passed through (number - 1) so callers can skip them."""
    if channel_number is None:
        return None
    if channel_number - 1 in channel_order:
        return channel_order.index(channel_number - 1)
    return channel_number - 1


def create_folder_for_sample(path, sample_name):
    folder_name = os.path.join(path, sample_name)
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def get_sample_name(path):
    filename = os.path.split(path)[1]
    if filename.endswith('.tiff'):
        return filename[:-5]
    return filename[:-4] # ends in .tif


# ── Processing steps ──────────────────────────────────────────────────────────

def segment_layer_channels(list_layer_numpydata, layer_keys, dict_models, channel_labels):
    """Each layer channel (DAPI at index 0, then the layers in order) with its own model (dict_models[key]),
    as in segment_projected_stack.py; for the layers other than DAPI, also the segmentation keeping only
    what matches the dilated DAPI nuclei mask. Returns (list_segmentation, list_segmentation_match_nuclei,
    list_cell_props, mask_nuclei)."""
    list_segmentation = []
    list_segmentation_match_nuclei = []
    list_cell_props = []
    mask_nuclei = None
    for i_channel, key in enumerate(layer_keys):
        print('Segment channel ' + channel_labels[i_channel] + ' (model ' + LAYER_MODEL_FILES[key] + ')')
        segmentation, _, _ = dict_models[key].eval(list_layer_numpydata[i_channel], diameter=None, channels= [[0,0]])
        list_segmentation.append(segmentation)
        list_cell_props.append(get_props_per_cell(segmentation))
        if i_channel == 0:  # DAPI
            #Mask nuclei little dilated
            struct2 = ndimage.generate_binary_structure(2, 2)
            mask_nuclei = ndimage.binary_dilation(segmentation > 0, structure=struct2)
            list_segmentation_match_nuclei.append(None)
        else:
            list_segmentation_match_nuclei.append(np.where(mask_nuclei, segmentation, 0))

    return list_segmentation, list_segmentation_match_nuclei, list_cell_props, mask_nuclei


def segment_edu(img_edu, model_edu, channel_label, sample_name, folder_output_expression):
    """EdU segmentation (comment 8 of the reviewer): Cellpose 2 re-trained model (diameter=None, channels [0,0])
    on the percentile-normalized EdU channel. Saves in folder_output_expression, file names ending in
    ENDING_EDU_OUTPUTS: the normalized channel (model input), the labels (16-bit png) and the binary mask
    (positives = 255). Returns the labels."""
    print('Segment channel ' + channel_label + ' (model ' + EDU_MODEL_FILE + ')')
    img_edu_norm = functionPercNorm(np.single(img_edu))
    segmentation, _, _ = model_edu.eval(img_edu_norm, diameter=None, channels=[[0, 0]])
    binary = np.where(segmentation > 0, 255, 0).astype(np.uint8)

    os.makedirs(folder_output_expression, exist_ok=True)
    path_base = os.path.join(folder_output_expression, sample_name + '_' + channel_label)
    cv2.imwrite(path_base + '_normalized' + ENDING_EDU_OUTPUTS + '.png', np.uint8(img_edu_norm * 255))
    cv2.imwrite(path_base + '_segmentation' + ENDING_EDU_OUTPUTS + '.png', segmentation.astype(np.uint16))
    cv2.imwrite(path_base + '_binary' + ENDING_EDU_OUTPUTS + '.png', binary)
    print(f'{sample_name}: {len(np.unique(segmentation)) - 1} {channel_label} positive cells')
    return segmentation


def plot_bins_bar(edges, values, ylabel, title, path_to_save):
    """Bar plot of one value per Y bin, labeled by the bin start (as each panel of get_distribution_histograms_list)."""
    plt.figure(figsize=(8, 4))
    plt.bar([f'{edge:.0f}' for edge in edges[:-1]], values)
    plt.xticks(rotation=90)
    plt.xlabel('Bin start (row, px)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300)
    plt.close()


def save_edu_histogram(segmentation_edu, dapi_cell_props, dapi_start_end, dapi_label, n_bins_edu, channel_label, sample_name,
                       folder_output_expression):
    """Distribution in Y of the EdU positives (comment 8): histogram of their centroids (get_density_bins) in
    n_bins_edu equal-height bins between the DAPI layer limits (dapi_start_end, with the start already overwritten
    in analyze_layers). Positives outside those limits are not counted. The DAPI cells (dapi_cell_props, of the DAPI
    segmentation) are counted in the same bins, to normalize: ratio = EdU positives / DAPI cells per bin (NaN if a bin
    has no DAPI cells). Saves in folder_output_expression, file names ending in ENDING_EDU_OUTPUTS: the csv with the
    counts and the ratio per bin, the histogram of the positives and the normalized one."""
    start_row, end_row = dapi_start_end
    if math.isnan(start_row) or math.isnan(end_row):
        print(f'{sample_name}: no {dapi_label} layer limits, no {channel_label} histogram')
        return
    dims = segmentation_edu.shape
    count, edges = get_density_bins(get_props_per_cell(segmentation_edu), dims[1], dims[0], axis=1, n_bins=n_bins_edu,
                                    range_start=start_row, range_end=end_row)
    count_dapi, _ = get_density_bins(dapi_cell_props, dims[1], dims[0], axis=1, n_bins=n_bins_edu,
                                     range_start=start_row, range_end=end_row)
    ratio = np.where(count_dapi > 0, count / np.maximum(count_dapi, 1), np.nan)

    path_base = os.path.join(folder_output_expression, sample_name + '_' + channel_label + '_histogram')
    pd.DataFrame({'bin': np.arange(1, n_bins_edu + 1), 'y_start': edges[:-1], 'y_end': edges[1:], 'n_positives': count,
                  'n_DAPI_cells': count_dapi, 'ratio_positives_to_DAPI': ratio}
                 ).to_csv(path_base + ENDING_EDU_OUTPUTS + '.csv', index=False)

    title = f'{sample_name} - between the {dapi_label} layer limits (rows {start_row}-{end_row})'
    plot_bins_bar(edges, count, 'Number of ' + channel_label + ' positive cells', title, path_base + ENDING_EDU_OUTPUTS + '.png')
    plot_bins_bar(edges, ratio, channel_label + ' positive cells / ' + dapi_label + ' cells', title,
                  path_base + '_normalized' + ENDING_EDU_OUTPUTS + '.png')
    print(f'{sample_name}: {count.sum()} {channel_label} positive cells and {count_dapi.sum()} {dapi_label} cells '
          f'between the {dapi_label} layer limits')


def analyze_layers(list_numpydata, list_segmentation, list_segmentation_match_nuclei, list_cell_props, mask_nuclei,
                   layer_keys, channel_labels, sample_name, folder_output_layers, config):
    """Layer delineation (Fig 5) of the layer channels (list_numpydata, in layer order with DAPI first, keys
    layer_keys): histograms, start/end row of each channel's layer, layer-filtered segmentations and their
    csv/png outputs. Returns (list_start_end, list_segmentation_filtered_layer)."""
    n_channels = len(list_numpydata)
    dims = list_numpydata[0].shape
    numpydata_C1_segmentation = list_segmentation[0]

    path_to_save = os.path.join(folder_output_layers, sample_name + '_nuclei_segmentation.png')
    plot_nuclei_segmentations_list(list_numpydata, list_segmentation, list_segmentation_match_nuclei,\
                                   path_to_save, mask_nuclei = mask_nuclei, channel_labels = channel_labels)

    path_to_save = os.path.join(folder_output_layers, sample_name + '_histograms.png')
    list_count = get_distribution_histograms_list(list_cell_props, dims, n_bins=config.n_bins, path_to_save = path_to_save, channel_labels = channel_labels)

    dict_counts = {channel_labels[i_channel]: list_count[i_channel] for i_channel in range(n_channels)}
    pd.DataFrame(dict_counts).to_csv(os.path.join(folder_output_layers, sample_name + '_bins.csv'), index=False, sep = ',')

    # Save nuclei segmentation
    for i_channel in range(n_channels):
        label = channel_labels[i_channel]
        cv2.imwrite(os.path.join(folder_output_layers, sample_name + '_' + label + '_segmentation.png'), list_segmentation[i_channel])
        if i_channel > 0:
            cv2.imwrite(os.path.join(folder_output_layers, sample_name + '_' + label + '_segmentation_masked.png'), list_segmentation_match_nuclei[i_channel])

    # Analyzing neural layer edge on the top

    # min_cells_bin per channel, as in segment_projected_stack.py: DAPI its own threshold, SATB2 (C2 in the
    # article) its own, BCL11B and TBR1 (C3, C4) share one
    min_cells_bin_by_key = {'DAPI': config.min_cells_bin_dapi, 'SATB2': config.min_cells_bin_filter_layer_2}
    min_cells_bin_per_channel = [min_cells_bin_by_key.get(key, config.min_cells_bin_filter_layer_3_4) for key in layer_keys]

    # For DAPI use its own (unmasked) segmentation; for the rest, use the nuclei-matched one
    list_segmentation_for_histogram = [numpydata_C1_segmentation] + list_segmentation_match_nuclei[1:]

    list_start_end = []
    for i_channel in range(n_channels):
        _, start_row, end_row = get_layer_nuclei_histogram(list_segmentation_for_histogram[i_channel], list_count[i_channel],\
                                                             min_cells_bin = min_cells_bin_per_channel[i_channel])
        list_start_end.append([start_row, end_row])

    #IMPORTANT: TO COMPUTE METRICS AND REMOVE CELLS IN THE TOP, NUCLEI STARTS WHERE C2 STARTS
    C2_start_row = list_start_end[1][0]
    if not math.isnan(C2_start_row):
        list_start_end[0][0] = np.min([start_row for start_row, end_row in list_start_end[1:]])

    # Filter segmentation according to the start and end rows
    list_segmentation_filtered_layer = []
    for i_channel in range(n_channels):
        start_row, end_row = list_start_end[i_channel]
        segmentation_source = numpydata_C1_segmentation if i_channel == 0 else list_segmentation_match_nuclei[i_channel]
        list_segmentation_filtered_layer.append(get_segmentation_filtered_layer(segmentation_source, start_row, end_row))

    list_number_cells_filtered_layer = [get_number_of_cells(seg) for seg in list_segmentation_filtered_layer]
    list_thickness = [end_row - start_row + 1 for start_row, end_row in list_start_end]

    #For the normalization, take the maximum of end between the last two layer channels
    #(the deep layers, BCL11B and TBR1, when both are there)
    bottom_row = max(end_row for start_row, end_row in list_start_end[-2:])
    C1_start_row = list_start_end[0][0]
    thickness_valid = bottom_row - C1_start_row + 1

    row_labels = ['n_cells_layer', 'start_row', 'end_row', 'thickness', 'thickness_sample', 'thickness_normalized', 'start_row_to_C1', 'end_row_to_C1']
    filtered_layer_data = {'--': row_labels}
    for i_channel in range(n_channels):
        start_row, end_row = list_start_end[i_channel]
        thickness = list_thickness[i_channel]
        filtered_layer_data[channel_labels[i_channel]] = [
            list_number_cells_filtered_layer[i_channel], start_row, end_row, thickness, thickness_valid,\
            thickness / thickness_valid, start_row - C1_start_row, end_row - C1_start_row]
    pd.DataFrame(filtered_layer_data).to_csv(os.path.join(folder_output_layers, sample_name + '_layers_filtered_by_bins.csv'), index=False, sep = ',')

    for i_channel in range(n_channels):
        cv2.imwrite(os.path.join(folder_output_layers, sample_name + '_' + channel_labels[i_channel] + '_segmentation_filtered_by_bins.png'),\
                    list_segmentation_filtered_layer[i_channel])

    path_to_save = os.path.join(folder_output_layers, sample_name + '_nuclei_segmentation_filtered_by_bins.png')
    plot_nuclei_segmentations_list(list_numpydata, list_segmentation_filtered_layer, list_segmentation_filtered_layer,\
                                   path_to_save, mask_nuclei = mask_nuclei, channel_labels = channel_labels)

    #Pairwise AND between all the layer channels except DAPI
    for c_1, c_2 in combinations(range(1, n_channels), 2):
        and_mask = np.uint8(np.logical_and((list_segmentation_filtered_layer[c_1] > 0).astype(np.uint8),\
                                            (list_segmentation_filtered_layer[c_2] > 0).astype(np.uint8)))
        and_number_cells = get_number_of_cells(and_mask)
        and_output = os.path.join(folder_output_layers, sample_name + '_' + channel_labels[c_1] + '_and_' + channel_labels[c_2] + '_nCells_' + str(and_number_cells) + '.png')
        cv2.imwrite(and_output, and_mask)

    # Layer delineation figure: segmentations plus start and end of each layer (no edge fitting, see segment_projected_stack.py)
    path_to_save = os.path.join(folder_output_layers, sample_name + '_layer_delineation.png')
    list_filtered_bool = [None] + [seg > 0 for seg in list_segmentation_filtered_layer[1:]]
    fig, ax = plot_nuclei_segmentations_list(list_numpydata, list_segmentation, list_filtered_bool,\
                                              path_to_save = None, mask_nuclei = mask_nuclei, channel_labels = channel_labels)
    x = [0, dims[1]-2]
    for i_channel in range(n_channels):
        start_row, end_row = list_start_end[i_channel]
        axis = get_layer_axis(ax, i_channel)
        axis.plot(x, [start_row, start_row], color='cyan', linewidth=config.marker_size/10)
        axis.plot(x, [end_row, end_row], color='cyan', linewidth=config.marker_size/10)
    plt.savefig(path_to_save, dpi=400)

    analyze_edge_fitting(list_numpydata, list_segmentation, list_segmentation_filtered_layer, list_start_end, mask_nuclei,
                         channel_labels, sample_name, folder_output_layers, config)

    return list_start_end, list_segmentation_filtered_layer


def get_layer_axis(ax, i_channel):
    """Panel of the plot_nuclei_segmentations_list figure where the layer limits / edges of channel i_channel are
    drawn (as in segment_projected_stack.py): DAPI's mask for Cs, the masked segmentation of the other channels."""
    base = i_channel * 3
    target = base + 2 if i_channel == 0 else base + 3
    return ax[(target - 1) // 6, (target - 1) % 6]


def analyze_edge_fitting(list_numpydata, list_segmentation, list_segmentation_filtered_layer, list_start_end, mask_nuclei,
                         channel_labels, sample_name, folder_output_layers, config):
    """Edge fitting (Fig 5) as in segment_projected_stack.py, to study the roughness of the layer edges: the layer-filtered
    segmentation of each layer channel is cut in vertical strips of config.subimage_width px, the extreme cell of each strip
    (top-most, and bottom-most) is selected and a line is fitted to those cells (top edge, and bottom edge). Saves in
    folder_output_layers: _edge_fitting.png (layer limits, extreme cells, fitted lines and their std error),
    _top_edge_fitting.csv and _bottom_edge_fitting.csv (fitting of each channel), and _top_edge_std_errors.csv and
    _top_edge_r_value.csv (top fitting with strip widths from 50 to 120 px)."""
    n_channels = len(list_numpydata)
    dims = list_numpydata[0].shape

    # Extreme cells per strip; the bottom ones only if there are top ones
    list_top_cells_xy = []
    list_bottom_cells_xy = []
    list_no_cells = []
    for segmentation_filtered in list_segmentation_filtered_layer:
        top_cell_labels, top_cells_xy = get_top_cells_labels(segmentation_filtered, subimage_width = config.subimage_width)
        no_cells = len(top_cell_labels) < 1
        bottom_cells_xy = None
        if not no_cells:
            bottom_cell_labels, bottom_cells_xy = get_top_cells_labels(segmentation_filtered, subimage_width = config.subimage_width,
                                                                       bottom_cells = True)
        list_top_cells_xy.append(top_cells_xy)
        list_bottom_cells_xy.append(bottom_cells_xy)
        list_no_cells.append(no_cells)

    # Linear fitting: (slope, intercept, r_value, p_value, std_err) per channel
    list_top_fit = [fit_cells(cells_xy) for cells_xy in list_top_cells_xy]
    list_bottom_fit = [fit_cells(cells_xy) for cells_xy in list_bottom_cells_xy]

    for edge, list_fit in [('bottom', list_bottom_fit), ('top', list_top_fit)]:
        data_fitting = {
                'channel': channel_labels,
                'r_value': [fit[2] for fit in list_fit],
                'p_value': [fit[3] for fit in list_fit],
                'std_error': [fit[4] for fit in list_fit],
                'slope': [fit[0] for fit in list_fit],
                'intercept': [fit[1] for fit in list_fit]
        }
        pd.DataFrame(data_fitting).to_csv(os.path.join(folder_output_layers, sample_name + '_' + edge + '_edge_fitting.csv'), index=False, sep = ',')

    # Different errors for different widths (top edge)
    subimage_widths, list_list_std_err, list_list_r_value = get_different_fittings_histogram_list(list_segmentation_filtered_layer)
    for list_list_values, ending in [(list_list_std_err, '_top_edge_std_errors.csv'), (list_list_r_value, '_top_edge_r_value.csv')]:
        fitting_top_cells_table = {'width': channel_labels}
        for i_width in range(len(subimage_widths)):
            fitting_top_cells_table[str(subimage_widths[i_width])] = [list_values[i_width] for list_values in list_list_values]
        pd.DataFrame(fitting_top_cells_table).to_csv(os.path.join(folder_output_layers, sample_name + ending), index=False, sep = ',')

    # Figure: layer limits, extreme cells (top and bottom) and fitted lines with their std error
    path_to_save = os.path.join(folder_output_layers, sample_name + '_edge_fitting.png')
    list_filtered_bool = [None] + [seg > 0 for seg in list_segmentation_filtered_layer[1:]]
    fig, ax = plot_nuclei_segmentations_list(list_numpydata, list_segmentation, list_filtered_bool,\
                                              path_to_save = None, mask_nuclei = mask_nuclei, channel_labels = channel_labels)
    x = [0, dims[1]-2]
    for i_channel in range(n_channels):
        if list_no_cells[i_channel]:
            continue
        axis = get_layer_axis(ax, i_channel)
        start_row, end_row = list_start_end[i_channel]
        axis.plot(x, [start_row, start_row], color='cyan', linewidth=config.marker_size/10)
        axis.plot(x, [end_row, end_row], color='cyan', linewidth=config.marker_size/10)
        plot_cells(axis, list_top_cells_xy[i_channel], marker_size = config.marker_size)
        plot_cells(axis, list_bottom_cells_xy[i_channel], color='green', marker_size = config.marker_size)

        top_slope, top_intercept, _, _, top_std_err = list_top_fit[i_channel]
        bottom_slope, bottom_intercept, _, _, bottom_std_err = list_bottom_fit[i_channel]
        top_y = top_slope*np.array(x) + top_intercept
        bottom_y = bottom_slope*np.array(x) + bottom_intercept
        axis.plot(x, top_y, color=config.fitting_color, label='Fitted line', linewidth=config.marker_size/10)
        axis.plot(x, bottom_y, color=config.fitting_color, label='Fitted line', linewidth=config.marker_size/10)
        if config.flag_text:
            axis.text(x[0] + 10, top_y[0] - config.text_shift, f'std-err: {top_std_err:.3f}', color=config.fitting_color, fontsize=config.fontsize)
            axis.text(x[0] + 10, bottom_y[0] - config.text_shift, f'std-err: {bottom_std_err:.3f}', color=config.fitting_color, fontsize=config.fontsize)
    fig.savefig(path_to_save, dpi=400)
    plt.close(fig)


def get_folder_output_expression(folder_output, sample_name, config):
    """(folder of the sample's expression (Fig 3) outputs, ending of their file names)."""
    ending_norm_expression = config.output_folder_expression + '_normalizedChannel' if config.flag_normalization_expression else ''
    return os.path.join(folder_output, sample_name + ending_norm_expression), ending_norm_expression


def analyze_joint_expression(list_numpydata, list_segmentation_filtered_layer, list_start_end, channel_order, channel_labels,
                             sample_name, folder_output, config):
    """Joint expression (Fig 3) of config.channel_1/2 plus the binned expression of config.channels_binned_expression,
    using the layer limits (list_start_end) and layer-filtered segmentations of analyze_layers (Fig 5).
    list_numpydata: all the channels, unnormalized, in the internal order (layer channels first); they are
    percentile-normalized here for the expression. Returns the list of per-channel binned-expression
    dataframes of this sample."""
    # 2026 09 - To quantify the joint expression, the channels are percentile-normalized:
    if config.flag_normalization_expression:
        list_channels = [functionPercNorm(np.single(ch)) for ch in list_numpydata]
    else:
        list_channels = [functionLinearNorm(ch) for ch in list_numpydata]

    folder_output_expression, ending_norm_expression = get_folder_output_expression(folder_output, sample_name, config)
    os.makedirs(folder_output_expression, exist_ok=True)

    for i_channel, img_channel in enumerate(list_channels):
        cv2.imwrite(os.path.join(folder_output_expression, channel_labels[i_channel] + ending_norm_expression + '.png'), np.uint8(img_channel * 255))

    # Viewer channel numbers (starting at 1) -> indices into the reordered (layer channels first) list_channels
    c_1 = to_internal_index(config.channel_1_number, channel_order)
    c_2 = to_internal_index(config.channel_2_number, channel_order)
    channels_binned_expr_idx = [(to_internal_index(number, channel_order), name, n_bins_channel)
                                for number, name, n_bins_channel in config.channels_binned_expression]

    # Channels without layer segmentation (internal index >= number of layer channels, e.g. EdU or an mRNA):
    # - empty segmentation, so they are never predicted positive (their Pred_ column is 0 and they don't add
    #   cells to the "only positives" plot; if both channels are like this, that plot is empty)
    # - their limits are not used: the "both layers" range comes from the layer channel(s) of the pair only,
    #   or from DAPI's layer if neither of the two is a layer channel
    n_layer_channels = len(list_start_end)
    empty_segmentation = np.zeros_like(list_segmentation_filtered_layer[0])
    list_segmentation_filtered_all = list_segmentation_filtered_layer + [empty_segmentation] * (len(list_channels) - n_layer_channels)
    list_limits = [list_start_end[c] for c in (c_1, c_2) if c < n_layer_channels]
    if not list_limits:
        print(f'{sample_name}: neither {config.channel_1_name} nor {config.channel_2_name} has layer segmentation, '
              f'using the layer limits of {channel_labels[0]}')
        list_limits = [list_start_end[0]]
    c_1_limits, c_2_limits = list_limits[0], list_limits[-1]

    return plot_generate_joint_expression_data(list_channels, list_segmentation_filtered_all, list_segmentation_filtered_layer[0], c_1 = c_1, c_2 = c_2,\
                                               c_1_limits = c_1_limits, c_2_limits = c_2_limits,\
                                               name_1 = config.channel_1_name, name_2 = config.channel_2_name,\
                                               ending_norm_expression = ending_norm_expression, sample_name = sample_name, folder_output_expression = folder_output_expression,\
                                               channels_binned_expr = channels_binned_expr_idx,\
                                               ratio_bin_min = config.ratio_bin_min, ratio_bin_max = config.ratio_bin_max, ratio_n_bins = config.ratio_n_bins,\
                                               dash_length_px = config.dash_length_px, gap_length_px = config.gap_length_px)


def process_sample(tiff_name, treatment, config, dict_models):
    """Whole per-sample pipeline: segmentation of the layer channels (one model each, dict_models[layer key]) and
    layer delineation (Fig 5), then the expression (Fig 3) with those same layers, and the EdU segmentation
    (dict_models[EDU_CHANNEL_NAME]) if a channel is named EdU. Returns its binned-expression dataframes (with
    'sample'/'treatment' columns), or [] if the sample is skipped."""
    tiff_path = os.path.join(config.folder_root, tiff_name)
    plt.close('all')
    sample_name = get_sample_name(tiff_path)
    print('Processing ' + sample_name + ' (' + treatment + ')')

    folder_output = create_folder_for_sample(config.folder_root, sample_name + config.output_folder_ending)
    folder_output_layers = os.path.join(folder_output, sample_name + config.output_folder_layers)
    os.makedirs(folder_output_layers, exist_ok=True)

    list_numpydata = read_tiff_channels(tiff_path)
    n_channels = len(list_numpydata)
    list_numpydata, channel_order, channel_labels, layer_keys = order_channels(list_numpydata, config.channel_names)

    # DAPI plus at least one other layer channel (to delineate the layers), and the two joint expression
    # channels, must be in this image
    if 'DAPI' not in layer_keys or len(layer_keys) < 2:
        print(f'{sample_name}: layer channels found {layer_keys}, needs DAPI and at least one of '
              f'{", ".join(list(LAYER_MODEL_FILES)[1:])}, skipping')
        return []
    for number in (config.channel_1_number, config.channel_2_number):
        if number > n_channels:
            print(f'{sample_name}: joint expression channel {number} ({config.channel_names[number]}) but there are {n_channels} channels, skipping')
            return []

    # Layer channels (the first len(layer_keys)), segmented unnormalized unless flag_normalize
    n_layer_channels = len(layer_keys)
    list_layer_numpydata = list_numpydata[:n_layer_channels]
    if config.flag_normalize:
        list_layer_numpydata = [functionPercNorm(np.single(ch)) for ch in list_layer_numpydata]

    list_segmentation, list_segmentation_match_nuclei, list_cell_props, mask_nuclei = \
        segment_layer_channels(list_layer_numpydata, layer_keys, dict_models, channel_labels)

    list_start_end, list_segmentation_filtered_layer = analyze_layers(list_layer_numpydata, list_segmentation, list_segmentation_match_nuclei,
                                                                      list_cell_props, mask_nuclei, layer_keys, channel_labels[:n_layer_channels],
                                                                      sample_name, folder_output_layers, config)

    list_df_binned_expr = analyze_joint_expression(list_numpydata, list_segmentation_filtered_layer, list_start_end, channel_order,
                                                   channel_labels, sample_name, folder_output, config)

    # EdU (comment 8): its own model on the percentile-normalized channel (whatever flag_normalize is),
    # outputs in the expression folder
    for edu_number in get_edu_channel_numbers(config.channel_names):
        if edu_number > n_channels:
            print(f'{sample_name}: EdU channel {edu_number} but there are {n_channels} channels, no EdU segmentation')
            continue
        i_edu = to_internal_index(edu_number, channel_order)
        folder_output_expression, _ = get_folder_output_expression(folder_output, sample_name, config)
        segmentation_edu = segment_edu(list_numpydata[i_edu], dict_models[EDU_CHANNEL_NAME], channel_labels[i_edu],
                                       sample_name, folder_output_expression)
        save_edu_histogram(segmentation_edu, list_cell_props[0], list_start_end[0], channel_labels[0], config.n_bins_edu,
                           channel_labels[i_edu], sample_name, folder_output_expression)

    for df_binned_expr in list_df_binned_expr:
        df_binned_expr.insert(0, 'treatment', treatment)
        df_binned_expr.insert(0, 'sample', sample_name)
    return list_df_binned_expr


def run_figure_3(config):
    """Runs every sample in config and then the aggregation across samples (boxplots per treatment and
    channel, saved in config.folder_root)."""
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    validate_config(config)

    # One model per layer channel used (and the EdU one if a channel is named EdU), loaded once and reused for every sample
    dict_models = {}
    for key in get_used_layer_keys(config.channel_names):
        print('Loading model ' + key + ': ' + get_layer_model_path(config.folder_models, key))
        dict_models[key] = models.CellposeModel(pretrained_model=get_layer_model_path(config.folder_models, key), gpu=config.flag_gpu)
    if get_edu_channel_numbers(config.channel_names):
        print('Loading model EdU: ' + get_edu_model_path(config.folder_models))
        dict_models[EDU_CHANNEL_NAME] = models.CellposeModel(pretrained_model=get_edu_model_path(config.folder_models), gpu=config.flag_gpu)

    list_df_binned_all = []  # binned expression of every sample, aggregated after the loop
    for tiff_name, treatment in zip(config.tiffs_name, config.tiffs_treatment):
        list_df_binned_all.extend(process_sample(tiff_name, treatment, config, dict_models))
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    if list_df_binned_all:
        df_binned_all = pd.concat(list_df_binned_all, ignore_index=True)
        df_binned_all.to_csv(os.path.join(config.folder_root, 'binned_expression_all_samples.csv'), index=False)
        plot_binned_expression_boxplots(df_binned_all, config.folder_root, config.channels_binned_expression)
    print('Done.')


def main():
    run_figure_3(build_config_from_parameters())


if __name__ == "__main__":
    main()
