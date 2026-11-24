#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################   PARAMETERS   #########################

folder_root = ''
tiffs_name = ['','']
tiffs_treatment = ['KO', 'WT']
channel_1_number, channel_1_name = 3, 'Tbr1'
channel_2_number, channel_2_name = 4, 'Bcl11b'
channel_3_number, channel_3_name = 2, 'EdU' # No data

# # Channels are numbered as in the image viewer, starting at 1 (C1 = DAPI). The names are used in
# # plot labels and output file/csv names, so you can tell where each result comes from.
# # Joint expression (Fig 3): scatter plots of channel 1 (X axis) vs channel 2 (Y axis), and the
# # Fig 3F intensity ratio log2(channel 2 / channel 1).
# channel_1_number, channel_1_name = 3, 'Tbr1'
# channel_2_number, channel_2_name = 4, 'Bcl11b'
# channel_3_number, channel_3_name = 2, 'EdU'

# Number of channels: each tiff is a multipage Z-max-projection, so every page in the
# multipage tiff is already one channel (no Z depth left to de-interleave). n_channels is
# therefore inferred per file from the number of pages - it is not a fixed global anymore,
# so a tiff can have 3 channels (C1, C2, C3) or 4 (C1, C2, C3, C4).

# Folder with the trained architectures. It has the subfolders Models_for_layer_delineation (one
# model per layer channel, see LAYER_MODEL_FILES below: DAPI, SATB2, BCL11B, TBR1, MRNA-BCL11B, MRNA-TBR1)
# and Models_EdU (model of the EdU positive cells, see POSITIVES_MODEL_FILES below). The model is chosen by
# the channel's name, so name the channels DAPI, SATB2, BCL11B, TBR1, MRNA-BCL11B, MRNA-TBR1, EdU (any case).
folder_models = ''

# Joint expression (Fig 3): pairs of channels (X axis, Y axis), numbered as in the image viewer; any named
# channel but DAPI. Each pair has all its outputs: scatter plots (all cells, only positives, cells in both
# layers and their top/bottom halves - Fig 3K), Fig 3F intensity ratio log2(Y / X) and rgb images.
pairs_joint_expression = [(channel_1_number, channel_2_number)]

# For expression (Fig 3)
flag_normalization_expression = True
# Expression of each cell (Fig 3: scatters, Fig 3F ratio, binned expression): median of its pixels (True, robust to
# speckle noise) or their mean (False, captures punctate expression that the median misses)
flag_use_median = True
output_folder_ending        = '_20261002'
output_folder_expression    = '_figure3_20261002'
output_folder_layers        = '_figure5_20261002'

# DAPI (nuclei reference) channel, numbered as in the image viewer starting at 1 - usually 1, but
# change it if the images were scanned in another order. Its name must be DAPI (that is how its model
# is found). Internally the layer channels are reordered (DAPI, SATB2, BCL11B, TBR1), but all
# labels/file names keep the original channel numbers (C1, C2, ...) as in the viewer.
channel_DAPI_number = 1
channel_DAPI_name = 'DAPI'


# Binned expression (Fig 3D), independent of the joint expression pairs: (channel number starting at 1, name, number
# of bins, layer channel number 1, layer channel number 2 or None). [top_y, bottom_y] goes from the top-most start to
# the bottom-most end of the layers of the layer channel(s) (DAPI, SATB2, BCL11B, TBR1, mRNA_BCL11B, mRNA_TBR1: the
# ones with layer delineation); with only one (the second None), its own layer - e.g. the channel's own layer, or
# DAPI's. That range is divided into that many equal-height Y bins, and the per-bin mean/median expression of the
# channel is computed in the DAPI cells of that range. Each row is saved to its own csv per sample, then aggregated
# across samples. A channel number of None (e.g. no EdU staining in these tiffs) is skipped.
channels_binned_expression = [(channel_2_number, channel_2_name, 4, channel_1_number, channel_2_number),
                              (channel_3_number, channel_3_name, 10, channel_1_number, channel_2_number)]

#Number of bins for histogram used for layers delinations (Fig 5)
n_bins = 20

# EdU (comment 8): number of bins of the histogram of the EdU positives, between the DAPI layer limits
n_bins_edu = 20

# Fig 3F: per-cell Y/X intensity ratio (log2) histogram of each pair of pairs_joint_expression (e.g. Bcl11b/Tbr1),
# one sample at a time for now (samples will later be combined into a single multi-age plot, like the reference figure).
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

#Parameters for running the segmentation (False: each layer channel normalized or not as in LAYER_MODEL_FILES - the
#proteins without normalizing, as in segment_projected_stack.py, the mRNAs normalized; True: all of them normalized)
flag_normalize = False
flag_gpu = True

# Dashed line (top_y/bottom_y markers) drawn on the rgb_with_lines debug image - run once,
# then check the dash spacing in the output image and adjust these two.
dash_length_px = 10
gap_length_px = 6

############################################################################

import os
import sys
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
        get_top_cells_labels, fit_cells, plot_cells, get_different_fittings_histogram_list, keep_sample_table,\
        get_binned_expression_stats, get_top_bottom_y


# ── Layer segmentation models ─────────────────────────────────────────────────

# Each layer channel is segmented with its own Cellpose model (the 4 architectures of the article, as in
# segment_projected_stack.py, plus the mRNA one), chosen by the channel's name (any case): {name: (model file in
# <folder_models>/Models_for_layer_delineation/, percentile-normalize the channel (functionPercNorm) before its
# segmentation)}. The proteins are segmented unnormalized (as in segment_projected_stack.py), the mRNAs normalized;
# flag_normalize normalizes all of them. Several names can share a model (both mRNAs). The order
# matters: DAPI first, then the layers from top (SATB2, upper layers) to bottom (BCL11B, TBR1, deep layers),
# then the mRNAs (of the deep layer markers) - the layer channels are processed in this order, which the layer
# delineation (analyze_layers) relies on. The mRNA channels are treated as the protein ones (BCL11B, TBR1):
# matched to the DAPI nuclei, layer delineation with min_cells_bin_filter_layer_3_4, edge fitting, joint expression.
# A channel named EdU is segmented with its own model (POSITIVES_MODEL_FILES below). Channels with any other
# name have no segmentation: they can still be quantified in the DAPI cells (joint and binned expression).
SUBFOLDER_LAYER_MODELS = 'Models_for_layer_delineation'
MRNA_MODEL_FILE = 'mRNA_both_model_nuclei_diam_8_ji_0.3538.311055'
LAYER_MODEL_FILES = {
    'DAPI':        ('Neurons_C1_DAPI.183326',   False),
    'SATB2':       ('Neurons_C2_SATB2.919883',  False),
    'BCL11B':      ('Neurons_C3_BCL11B.981474', False),
    'TBR1':        ('Neurons_C4_TBR1.909737',   False),
    'MRNA-BCL11B': (MRNA_MODEL_FILE,            True),
    'MRNA-TBR1':   (MRNA_MODEL_FILE,            True),
}

# Positive cells (EdU, comment 8 of the reviewer): a channel named as one of the keys of POSITIVES_MODEL_FILES
# (any case) is segmented with that model (Cellpose 2 re-trained, diameter=None), on the percentile-normalized
# channel: {name: model file in <folder_models>/Models_EdU/}. It is not a layer channel (no layer delineation).
# Its outputs go to the expression (Fig 3) folder, with the file names ending in ENDING_EDU_OUTPUTS. In the joint
# expression, its segmentation gives which cells are positive (Pred_ column).
SUBFOLDER_EDU_MODEL = 'Models_EdU'
EDU_MODEL_FILE = 'model_positives_EdU.253538'
POSITIVES_MODEL_FILES = {
    'EdU': EDU_MODEL_FILE,
}
ENDING_EDU_OUTPUTS = '_comment8'

def get_layer_key(name):
    """Key of LAYER_MODEL_FILES ('DAPI', 'SATB2', 'BCL11B', 'TBR1', 'MRNA-BCL11B', 'MRNA-TBR1') if name is one of them (any case), else None."""
    if not name:
        return None
    key = name.strip().upper()
    return key if key in LAYER_MODEL_FILES else None


def get_layer_model_file(layer_key):
    return LAYER_MODEL_FILES[layer_key][0]


def is_layer_channel_normalized(layer_key):
    """True if the layer channel is percentile-normalized (functionPercNorm) before its segmentation (LAYER_MODEL_FILES)."""
    return LAYER_MODEL_FILES[layer_key][1]


def get_layer_model_path(folder_models, layer_key):
    return os.path.join(folder_models, SUBFOLDER_LAYER_MODELS, get_layer_model_file(layer_key))


def get_used_layer_keys(channel_names):
    """Layer keys of the named channels ({number: name}), in layer order."""
    keys = {get_layer_key(name) for name in channel_names.values()}
    return [key for key in LAYER_MODEL_FILES if key in keys]


def get_positives_key(name):
    """Key of POSITIVES_MODEL_FILES (e.g. 'EdU') if name is one of them (any case), else None."""
    if not name:
        return None
    for key in POSITIVES_MODEL_FILES:
        if name.strip().upper() == key.upper():
            return key
    return None


def get_positives_model_path(folder_models, positives_key):
    return os.path.join(folder_models, SUBFOLDER_EDU_MODEL, POSITIVES_MODEL_FILES[positives_key])


def get_positives_channels(channel_names):
    """[(viewer number, positives key)] of the channels of {number: name} named as a POSITIVES_MODEL_FILES key,
    in the viewer's order (each key at most once, see validate_config)."""
    return [(number, get_positives_key(name)) for number, name in sorted(channel_names.items()) if get_positives_key(name)]


def get_pair_names(config):
    """[(X name, Y name)] of config.pairs_joint_expression."""
    return [(config.channel_names[number_1], config.channel_names[number_2]) for number_1, number_2 in config.pairs_joint_expression]


def get_binned_layers_name(config, layer_number_1, layer_number_2):
    """Name of the layers of a binned expression row (used in file names, titles and the 'layers' column): the
    name of its layer channel, or of its two layer channels joined with '_and_' (e.g. 'Tbr1_and_Bcl11b')."""
    return '_and_'.join(config.channel_names[number] for number in (layer_number_1, layer_number_2) if number is not None)


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
    pairs_joint_expression: list        # [(X viewer channel number, Y viewer channel number), ...]: joint expression X/Y axes
                                        # (Fig 3F ratio = Y / X); any named channels but DAPI
    channels_binned_expression: list    # [(viewer channel number, name, n_bins, layer channel number 1, layer channel
                                        # number 2 or None), ...]: binned in the range of the layer(s) of the layer channel(s)
    folder_models: str                  # folder with the Models_for_layer_delineation and Models_EdU subfolders

    # Optional parameters (defaults = PARAMETERS section above)
    flag_normalization_expression: bool = flag_normalization_expression
    flag_use_median: bool = flag_use_median
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


# Optional parameters shown (and editable) in the GUI's second tab: (field name, description).
OPTIONAL_PARAMETERS = [
    ('flag_normalization_expression', 'Percentile-normalize channels before quantifying expression (Fig 3)'),
    ('flag_use_median',               'Expression of each cell (Fig 3): median of its pixels (off: mean, captures punctate expression)'),
    ('output_folder_ending',          'Suffix of each sample output folder'),
    ('output_folder_expression',      'Suffix of the expression (Fig 3) outputs'),
    ('output_folder_layers',          'Suffix of the layers (Fig 5) output folder'),
    ('n_bins',                        'Number of bins of the histograms used for layer delineation (Fig 5)'),
    ('n_bins_edu',                    'Number of bins of the EdU positives histogram, between the DAPI layer limits (comment 8)'),
    ('ratio_bin_min',                 'Fig 3F ratio histogram (log2 Y/X of each pair): X axis minimum'),
    ('ratio_bin_max',                 'Fig 3F ratio histogram (log2 Y/X of each pair): X axis maximum'),
    ('ratio_n_bins',                  'Fig 3F ratio histogram (log2 Y/X of each pair): number of bins'),
    ('min_cells_bin_dapi',            'Layer delineation: min cells per bin, DAPI'),
    ('min_cells_bin_filter_layer_2',  'Layer delineation: min cells per bin, first channel after DAPI'),
    ('min_cells_bin_filter_layer_3_4','Layer delineation: min cells per bin, remaining channels (BCL11B, TBR1, MRNA-BCL11B, MRNA-TBR1)'),
    ('marker_size',                   'Line width (/10) of the layer limits and fitted lines, and size of the extreme cells (Fig 5 figures)'),
    ('subimage_width',                'Edge fitting: width (px) of the vertical strips where the extreme cell is selected'),
    ('flag_text',                     'Edge fitting: write the std error of each fitted line'),
    ('text_shift',                    'Edge fitting: how much above the fitted line the std error text goes (px)'),
    ('fitting_color',                 'Edge fitting: color of the fitted lines and text (matplotlib color name)'),
    ('fontsize',                      'Edge fitting: font size of the std error text'),
    ('flag_normalize',                'Percentile-normalize all the layer channels before their segmentation (off: only the mRNAs, as in LAYER_MODEL_FILES)'),
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
                         pairs_joint_expression=pairs_joint_expression,
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

    # Joint expression pairs: any named channels but DAPI. A non-layer channel (EdU, or a name without model)
    # has no layer limits, see analyze_joint_expression.
    for number_1, number_2 in config.pairs_joint_expression:
        for number in (number_1, number_2):
            name = config.channel_names.get(number)
            if not name or get_layer_key(name) == 'DAPI':
                raise ValueError(f'The joint expression channels must be named channels other than DAPI; channel {number} is "{name}".')
        if number_1 == number_2:
            raise ValueError(f'The two channels of a joint expression pair must be different: {config.channel_names[number_1]}.')
    if len(set(config.pairs_joint_expression)) != len(config.pairs_joint_expression):
        raise ValueError('Each joint expression pair (X, Y) can be used only once.')

    # Binned expression: its range comes from the layers of one or two layer channels (they have layer delineation)
    for number, name, n_bins_channel, layer_number_1, layer_number_2 in config.channels_binned_expression:
        if layer_number_1 is None:
            raise ValueError(f'Binned expression of {name}: choose the (first) layer channel whose layer gives the range of the bins.')
        for layer_number in (layer_number_1, layer_number_2):
            if layer_number is not None and not get_layer_key(config.channel_names.get(layer_number)):
                raise ValueError(f'Binned expression of {name}: the layers must be of layer channels ({", ".join(LAYER_MODEL_FILES)}); '
                                 f'channel {layer_number} is "{config.channel_names.get(layer_number)}".')
        if layer_number_1 == layer_number_2:
            raise ValueError(f'Binned expression of {name}: the two layer channels must be different (leave the second one empty to use only one).')
    rows_binned = [(number, layer_number_1, layer_number_2) for number, _, _, layer_number_1, layer_number_2 in config.channels_binned_expression]
    if len(set(rows_binned)) != len(rows_binned):
        raise ValueError('Each binned expression row (channel and layers) can be used only once.')

    positives_keys = [key for _, key in get_positives_channels(config.channel_names)]
    if len(positives_keys) != len(set(positives_keys)):
        raise ValueError(f'Each positives channel ({", ".join(POSITIVES_MODEL_FILES)}) can be used only once: {positives_keys}')

    if not config.folder_models:
        raise ValueError('Set folder_models (the folder with the ' + SUBFOLDER_LAYER_MODELS + ' and ' + SUBFOLDER_EDU_MODEL + ' subfolders).')
    paths_models = [get_layer_model_path(config.folder_models, key) for key in get_used_layer_keys(config.channel_names)]
    paths_models += [get_positives_model_path(config.folder_models, key) for key in positives_keys]
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
    TBR1, MRNA-BCL11B, MRNA-TBR1 - the ones present), then the other channels (e.g. EdU, unnamed) in the file's order.
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
        print('Segment channel ' + channel_labels[i_channel] + ' (model ' + get_layer_model_file(key) + ')')
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


def segment_positives(img_positives, model_positives, model_file, channel_label, sample_name, folder_output_expression):
    """Segmentation of the positive cells of a positives channel (EdU - comment 8 of the reviewer): Cellpose 2
    re-trained model (model_positives, loaded from model_file; diameter=None, channels [0,0]) on the percentile-normalized
    channel. Saves in folder_output_expression, file names ending in ENDING_EDU_OUTPUTS: the normalized channel (model
    input), the labels (16-bit png) and the binary mask (positives = 255). Returns the labels."""
    print('Segment channel ' + channel_label + ' (model ' + model_file + ')')
    img_positives_norm = functionPercNorm(np.single(img_positives))
    segmentation, _, _ = model_positives.eval(img_positives_norm, diameter=None, channels=[[0, 0]])
    binary = np.where(segmentation > 0, 255, 0).astype(np.uint8)

    os.makedirs(folder_output_expression, exist_ok=True)
    path_base = os.path.join(folder_output_expression, sample_name + '_' + channel_label)
    cv2.imwrite(path_base + '_normalized' + ENDING_EDU_OUTPUTS + '.png', np.uint8(img_positives_norm * 255))
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


def save_positives_histogram(segmentation_positives, dapi_cell_props, dapi_start_end, dapi_label, n_bins_edu, channel_label,
                             sample_name, folder_output_expression, dict_tables=None):
    """Distribution in Y of the positives of a positives channel (EdU - comment 8): histogram of their centroids
    (get_density_bins) in n_bins_edu equal-height bins between the DAPI layer limits (dapi_start_end, with the start
    already overwritten in analyze_layers). Positives outside those limits are not counted. The DAPI cells
    (dapi_cell_props, of the DAPI segmentation) are counted in the same bins, to normalize: ratio = positives / DAPI cells
    per bin (NaN if a bin has no DAPI cells). Saves in folder_output_expression, file names ending in ENDING_EDU_OUTPUTS:
    the csv with the counts and the ratio per bin (also kept in dict_tables, see keep_sample_table), the histogram
    of the positives and the normalized one."""
    start_row, end_row = dapi_start_end
    if math.isnan(start_row) or math.isnan(end_row):
        print(f'{sample_name}: no {dapi_label} layer limits, no {channel_label} histogram')
        return
    dims = segmentation_positives.shape
    count, edges = get_density_bins(get_props_per_cell(segmentation_positives), dims[1], dims[0], axis=1, n_bins=n_bins_edu,
                                    range_start=start_row, range_end=end_row)
    count_dapi, _ = get_density_bins(dapi_cell_props, dims[1], dims[0], axis=1, n_bins=n_bins_edu,
                                     range_start=start_row, range_end=end_row)
    ratio = np.where(count_dapi > 0, count / np.maximum(count_dapi, 1), np.nan)

    path_base = os.path.join(folder_output_expression, sample_name + '_' + channel_label + '_histogram')
    df_histogram = pd.DataFrame({'bin': np.arange(1, n_bins_edu + 1), 'y_start': edges[:-1], 'y_end': edges[1:], 'n_positives': count,
                                 'n_DAPI_cells': count_dapi, 'ratio_positives_to_DAPI': ratio})
    df_histogram.to_csv(path_base + ENDING_EDU_OUTPUTS + '.csv', index=False)
    keep_sample_table(dict_tables, path_base + ENDING_EDU_OUTPUTS + '.csv', sample_name, df_histogram)

    title = f'{sample_name} - between the {dapi_label} layer limits (rows {start_row}-{end_row})'
    plot_bins_bar(edges, count, 'Number of ' + channel_label + ' positive cells', title, path_base + ENDING_EDU_OUTPUTS + '.png')
    plot_bins_bar(edges, ratio, channel_label + ' positive cells / ' + dapi_label + ' cells', title,
                  path_base + '_normalized' + ENDING_EDU_OUTPUTS + '.png')
    print(f'{sample_name}: {count.sum()} {channel_label} positive cells and {count_dapi.sum()} {dapi_label} cells '
          f'between the {dapi_label} layer limits')


def analyze_layers(list_numpydata, list_segmentation, list_segmentation_match_nuclei, list_cell_props, mask_nuclei,
                   layer_keys, channel_labels, sample_name, folder_output_layers, config, dict_tables=None):
    """Layer delineation (Fig 5) of the layer channels (list_numpydata, in layer order with DAPI first, keys
    layer_keys): histograms, start/end row of each channel's layer, layer-filtered segmentations and their
    csv/png outputs (the csvs also kept in dict_tables, see keep_sample_table).
    Returns (list_start_end, list_segmentation_filtered_layer)."""
    n_channels = len(list_numpydata)
    dims = list_numpydata[0].shape
    numpydata_C1_segmentation = list_segmentation[0]

    path_to_save = os.path.join(folder_output_layers, sample_name + '_nuclei_segmentation.png')
    plot_nuclei_segmentations_list(list_numpydata, list_segmentation, list_segmentation_match_nuclei,\
                                   path_to_save, mask_nuclei = mask_nuclei, channel_labels = channel_labels)

    path_to_save = os.path.join(folder_output_layers, sample_name + '_histograms.png')
    list_count = get_distribution_histograms_list(list_cell_props, dims, n_bins=config.n_bins, path_to_save = path_to_save, channel_labels = channel_labels)

    dict_counts = {channel_labels[i_channel]: list_count[i_channel] for i_channel in range(n_channels)}
    df_bins = pd.DataFrame(dict_counts)
    path_csv = os.path.join(folder_output_layers, sample_name + '_bins.csv')
    df_bins.to_csv(path_csv, index=False, sep = ',')
    # Aggregated with the bin number (1 = top), so the rows of each sample can be told apart
    df_bins.insert(0, 'bin', np.arange(1, len(df_bins) + 1))
    keep_sample_table(dict_tables, path_csv, sample_name, df_bins)

    # Save nuclei segmentation
    for i_channel in range(n_channels):
        label = channel_labels[i_channel]
        cv2.imwrite(os.path.join(folder_output_layers, sample_name + '_' + label + '_segmentation.png'), list_segmentation[i_channel])
        if i_channel > 0:
            cv2.imwrite(os.path.join(folder_output_layers, sample_name + '_' + label + '_segmentation_masked.png'), list_segmentation_match_nuclei[i_channel])

    # Analyzing neural layer edge on the top

    # min_cells_bin per channel, as in segment_projected_stack.py: DAPI its own threshold, SATB2 (C2 in the
    # article) its own, BCL11B and TBR1 (C3, C4) share one, also used for the mRNAs (MRNA-BCL11B, MRNA-TBR1)
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
    #(the deep layers: BCL11B, TBR1 or their mRNAs, the last ones in LAYER_MODEL_FILES)
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
    df_layers = pd.DataFrame(filtered_layer_data)
    path_csv = os.path.join(folder_output_layers, sample_name + '_layers_filtered_by_bins.csv')
    df_layers.to_csv(path_csv, index=False, sep = ',')
    keep_sample_table(dict_tables, path_csv, sample_name, df_layers)

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
                         channel_labels, sample_name, folder_output_layers, config, dict_tables = dict_tables)

    return list_start_end, list_segmentation_filtered_layer


def get_layer_axis(ax, i_channel):
    """Panel of the plot_nuclei_segmentations_list figure where the layer limits / edges of channel i_channel are
    drawn (as in segment_projected_stack.py): DAPI's mask for Cs, the masked segmentation of the other channels."""
    base = i_channel * 3
    target = base + 2 if i_channel == 0 else base + 3
    return ax[(target - 1) // 6, (target - 1) % 6]


def analyze_edge_fitting(list_numpydata, list_segmentation, list_segmentation_filtered_layer, list_start_end, mask_nuclei,
                         channel_labels, sample_name, folder_output_layers, config, dict_tables=None):
    """Edge fitting (Fig 5) as in segment_projected_stack.py, to study the roughness of the layer edges: the layer-filtered
    segmentation of each layer channel is cut in vertical strips of config.subimage_width px, the extreme cell of each strip
    (top-most, and bottom-most) is selected and a line is fitted to those cells (top edge, and bottom edge). Saves in
    folder_output_layers: _edge_fitting.png (layer limits, extreme cells, fitted lines and their std error),
    _top_edge_fitting.csv and _bottom_edge_fitting.csv (fitting of each channel), and _top_edge_std_errors.csv and
    _top_edge_r_value.csv (top fitting with strip widths from 50 to 120 px). The csvs are also kept in dict_tables
    (see keep_sample_table)."""
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
        df_fitting = pd.DataFrame(data_fitting)
        path_csv = os.path.join(folder_output_layers, sample_name + '_' + edge + '_edge_fitting.csv')
        df_fitting.to_csv(path_csv, index=False, sep = ',')
        keep_sample_table(dict_tables, path_csv, sample_name, df_fitting)

    # Different errors for different widths (top edge)
    subimage_widths, list_list_std_err, list_list_r_value = get_different_fittings_histogram_list(list_segmentation_filtered_layer)
    for list_list_values, ending in [(list_list_std_err, '_top_edge_std_errors.csv'), (list_list_r_value, '_top_edge_r_value.csv')]:
        fitting_top_cells_table = {'width': channel_labels}
        for i_width in range(len(subimage_widths)):
            fitting_top_cells_table[str(subimage_widths[i_width])] = [list_values[i_width] for list_values in list_list_values]
        df_fitting_widths = pd.DataFrame(fitting_top_cells_table)
        path_csv = os.path.join(folder_output_layers, sample_name + ending)
        df_fitting_widths.to_csv(path_csv, index=False, sep = ',')
        keep_sample_table(dict_tables, path_csv, sample_name, df_fitting_widths)

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


# ── Parameters of the run (txt in every output folder) ────────────────────────

PARAMETERS_TXT_NAME = 'parameters_used.txt'


def describe_channel_segmentation(name, config):
    """How a channel named name is segmented in this run (for the parameters txt)."""
    layer_key = get_layer_key(name)
    if layer_key:
        normalized = config.flag_normalize or is_layer_channel_normalized(layer_key)
        return (f'layer channel, model {get_layer_model_file(layer_key)} '
                f'({"percentile-normalized" if normalized else "not normalized"} before its segmentation)')
    positives_key = get_positives_key(name)
    if positives_key:
        return f'positives (comment 8), model {POSITIVES_MODEL_FILES[positives_key]} (percentile-normalized before its segmentation)'
    return 'no segmentation'


def get_parameters_text(config):
    """Readable summary of everything a run uses (config, models and optional parameters), saved by
    save_parameters_txt in every output folder so the user can see with which parameters the outputs were made."""
    lines = ['Parameters used by script_analyze_layers.py',
             'Run started: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             '',
             'Images folder (folder_root): ' + config.folder_root,
             'Models folder (folder_models): ' + config.folder_models,
             '',
             'Images: treatment']
    lines += [f'  {tiff_name}: {treatment}' for tiff_name, treatment in zip(config.tiffs_name, config.tiffs_treatment)]
    lines += ['', 'Channels (numbered as in the image viewer, starting at 1): segmentation']
    lines += [f'  C{number} {name}: {describe_channel_segmentation(name, config)}' for number, name in sorted(config.channel_names.items())]
    lines += ['', 'Joint expression pairs, X vs Y (Fig 3F ratio = log2 Y/X):']
    lines += [f'  {name_1} (C{number_1}) vs {name_2} (C{number_2})'
              for (number_1, number_2), (name_1, name_2) in zip(config.pairs_joint_expression, get_pair_names(config))] or ['  none']
    lines += ['', 'Binned expression: number of bins, layers that give the range of the bins']
    lines += [f'  {name} (C{number}): {n_bins_channel} bins, in the layers of '
              f'{get_binned_layers_name(config, layer_number_1, layer_number_2).replace("_and_", " and ")}'
              for number, name, n_bins_channel, layer_number_1, layer_number_2 in config.channels_binned_expression] or ['  none']
    lines += ['', 'Optional parameters:']
    width = max(len(name) for name, _ in OPTIONAL_PARAMETERS)
    lines += [f'  {name.ljust(width)} = {getattr(config, name)}    # {description}' for name, description in OPTIONAL_PARAMETERS]
    return '\n'.join(lines) + '\n'


def save_parameters_txt(folder, parameters_text):
    """Saves parameters_text (get_parameters_text) as PARAMETERS_TXT_NAME in folder."""
    with open(os.path.join(folder, PARAMETERS_TXT_NAME), 'w') as f:
        f.write(parameters_text)


def analyze_joint_expression(list_numpydata, list_segmentation_filtered_layer, list_start_end, channel_order, channel_labels,
                             sample_name, folder_output, config, dict_segmentation_positives=None, dict_tables=None):
    """Expression (Fig 3): joint expression of each pair of config.pairs_joint_expression and binned expression of
    each row of config.channels_binned_expression (analyze_binned_expression), using the layer limits (list_start_end)
    and layer-filtered segmentations of analyze_layers (Fig 5). list_numpydata: all the channels, unnormalized, in the
    internal order (layer channels first); they are percentile-normalized here for the expression, whose value per
    cell is the median of its pixels (or the mean if not config.flag_use_median). dict_segmentation_positives:
    {internal index: segmentation} of the positives channels (EdU), used as their segmentation (Pred_ column). The
    csvs of the scatters and Fig 3F of each pair are kept in dict_tables (see keep_sample_table). Returns the list of
    binned-expression dataframes of this sample (one per row, with a 'layers' column)."""
    # 2026 09 - To quantify the joint expression, the channels are percentile-normalized:
    if config.flag_normalization_expression:
        list_channels = [functionPercNorm(np.single(ch)) for ch in list_numpydata]
    else:
        list_channels = [functionLinearNorm(ch) for ch in list_numpydata]

    folder_output_expression, ending_norm_expression = get_folder_output_expression(folder_output, sample_name, config)
    os.makedirs(folder_output_expression, exist_ok=True)

    for i_channel, img_channel in enumerate(list_channels):
        cv2.imwrite(os.path.join(folder_output_expression, channel_labels[i_channel] + ending_norm_expression + '.png'), np.uint8(img_channel * 255))

    # Channels without layer delineation (internal index >= number of layer channels: EdU, or names without model):
    # - EdU: its positives segmentation (whole image, not filtered by layer), so a cell is predicted EdU positive
    #   (Pred_ column) if it is mostly inside an EdU positive
    # - names without model: empty segmentation, so they are never predicted positive (their Pred_ column is 0 and
    #   they don't add cells to the "only positives" plot; if both channels are like this, that plot is empty)
    # - their limits are not used: the "both layers" range comes from the layer channel(s) of the pair only,
    #   or from DAPI's layer if neither of the two is a layer channel
    n_layer_channels = len(list_start_end)
    dict_segmentation_positives = dict_segmentation_positives or {}
    empty_segmentation = np.zeros_like(list_segmentation_filtered_layer[0])
    list_segmentation_filtered_all = list_segmentation_filtered_layer + [dict_segmentation_positives.get(i_channel, empty_segmentation)
                                                                         for i_channel in range(n_layer_channels, len(list_channels))]

    for (number_1, number_2), (name_1, name_2) in zip(config.pairs_joint_expression, get_pair_names(config)):
        if max(number_1, number_2) > len(list_channels):
            print(f'{sample_name}: joint expression pair {name_1} vs {name_2} needs channel {max(number_1, number_2)} but there '
                  f'are {len(list_channels)} channels, skipping the pair')
            continue
        c_1 = to_internal_index(number_1, channel_order)
        c_2 = to_internal_index(number_2, channel_order)

        list_limits = [list_start_end[c] for c in (c_1, c_2) if c < n_layer_channels]
        if not list_limits:
            print(f'{sample_name}: neither {name_1} nor {name_2} has layer segmentation, '
                  f'using the layer limits of {channel_labels[0]}')
            list_limits = [list_start_end[0]]
        c_1_limits, c_2_limits = list_limits[0], list_limits[-1]

        plot_generate_joint_expression_data(list_channels, list_segmentation_filtered_all, list_segmentation_filtered_layer[0], c_1 = c_1, c_2 = c_2,\
                                            c_1_limits = c_1_limits, c_2_limits = c_2_limits,\
                                            name_1 = name_1, name_2 = name_2,\
                                            ending_norm_expression = ending_norm_expression, sample_name = sample_name, folder_output_expression = folder_output_expression,\
                                            ratio_bin_min = config.ratio_bin_min, ratio_bin_max = config.ratio_bin_max, ratio_n_bins = config.ratio_n_bins,\
                                            dash_length_px = config.dash_length_px, gap_length_px = config.gap_length_px,\
                                            flag_use_median = config.flag_use_median, dict_tables = dict_tables)

    return analyze_binned_expression(list_channels, list_segmentation_filtered_layer[0], list_start_end, channel_order,
                                     sample_name, folder_output_expression, ending_norm_expression, config)


def analyze_binned_expression(list_channels, dapi_segmentation_filtered_layer, list_start_end, channel_order,
                              sample_name, folder_output_expression, ending_norm_expression, config):
    """Binned expression (Fig 3D) of each row of config.channels_binned_expression ((channel number, name, n_bins,
    layer channel number 1, layer channel number 2 or None), viewer numbers): [top_y, bottom_y] goes from the top-most
    start to the bottom-most end of the layer(s) of its layer channel(s) (list_start_end, get_top_bottom_y; with only
    one layer channel, its own layer), divided into n_bins equal-height Y bins; in each bin, the mean/median across
    cells of the per-cell expression of the channel (list_channels, already normalized; median or mean of the pixels
    of each cell, config.flag_use_median), in the DAPI cells (dapi_segmentation_filtered_layer, already filtered by
    DAPI's layer) of [top_y, bottom_y]. One csv per row in folder_output_expression. Returns their dataframes, with a
    'layers' column (get_binned_layers_name), to aggregate them across samples."""
    n_channels = len(list_channels)
    list_df_binned_expr = []
    for number, name, n_bins_channel, layer_number_1, layer_number_2 in config.channels_binned_expression:
        layers_name = get_binned_layers_name(config, layer_number_1, layer_number_2)
        layer_numbers = [n for n in (layer_number_1, layer_number_2) if n is not None]
        if number is None or max([number] + layer_numbers) > n_channels:
            print(f'{sample_name}: skipping the binned expression of {name} in the layers of {layers_name}: '
                  f'channel not available ({n_channels} channels)')
            continue

        # The layer channels are the first ones in the internal order, so their internal index is their list_start_end index
        list_limits = [list_start_end[to_internal_index(layer_number, channel_order)] for layer_number in layer_numbers]
        if any(math.isnan(limit) for limits in list_limits for limit in limits):
            print(f'{sample_name}: no layer limits of {layers_name}, skipping the binned expression of {name} in them')
            continue
        top_y, bottom_y = get_top_bottom_y(list_limits)

        cells_in_layers = get_segmentation_filtered_layer(dapi_segmentation_filtered_layer, top_y, bottom_y)
        df_binned_expr = get_binned_expression_stats(cells_in_layers, list_channels[to_internal_index(number, channel_order)], name,
                                                     top_y, bottom_y, n_bins_channel, flag_use_median = config.flag_use_median)
        path_csv = os.path.join(folder_output_expression,
                                f'{sample_name}_binned_expression_{name}_in_{layers_name}{ending_norm_expression}.csv')
        df_binned_expr.to_csv(path_csv, index=False)
        # The range of the bins depends on the layers, so they are kept to aggregate the rows
        df_binned_expr.insert(0, 'layers', layers_name)
        list_df_binned_expr.append(df_binned_expr)
    return list_df_binned_expr


def process_sample(tiff_name, treatment, config, dict_models, parameters_text):
    """Whole per-sample pipeline: segmentation of the layer channels (one model each, dict_models[layer key]) and
    layer delineation (Fig 5), the segmentation of the positives channels (EdU: dict_models[positives key]), then the
    expression (Fig 3) of each joint expression pair and binned expression row with those same layers (and the EdU
    positives). parameters_text (get_parameters_text) is saved in each output folder of the sample. Returns (its
    binned-expression dataframes, with 'sample'/'treatment'/'layers' columns; its other csvs, {file name without the sample name: table, with
    'sample'/'treatment' columns}, see keep_sample_table), or ([], {}) if the sample is skipped."""
    tiff_path = os.path.join(config.folder_root, tiff_name)
    plt.close('all')
    sample_name = get_sample_name(tiff_path)
    print('Processing ' + sample_name + ' (' + treatment + ')')

    folder_output = create_folder_for_sample(config.folder_root, sample_name + config.output_folder_ending)
    folder_output_layers = os.path.join(folder_output, sample_name + config.output_folder_layers)
    os.makedirs(folder_output_layers, exist_ok=True)
    save_parameters_txt(folder_output, parameters_text)
    save_parameters_txt(folder_output_layers, parameters_text)

    list_numpydata = read_tiff_channels(tiff_path)
    n_channels = len(list_numpydata)
    list_numpydata, channel_order, channel_labels, layer_keys = order_channels(list_numpydata, config.channel_names)

    # DAPI plus at least one other layer channel (to delineate the layers) must be in this image (a joint
    # expression pair with a channel missing in this image is skipped in analyze_joint_expression)
    if 'DAPI' not in layer_keys or len(layer_keys) < 2:
        print(f'{sample_name}: layer channels found {layer_keys}, needs DAPI and at least one of '
              f'{", ".join(list(LAYER_MODEL_FILES)[1:])}, skipping')
        return [], {}

    # Every csv of this sample (but the binned expression, aggregated apart): {file name without the sample name: table}
    dict_tables = {}

    # Expression (Fig 3) outputs folder (also the one of the positives - EdU - outputs)
    folder_output_expression, _ = get_folder_output_expression(folder_output, sample_name, config)
    os.makedirs(folder_output_expression, exist_ok=True)
    save_parameters_txt(folder_output_expression, parameters_text)

    # Layer channels (the first len(layer_keys)): each percentile-normalized before its segmentation if
    # LAYER_MODEL_FILES says so (the mRNAs) or if flag_normalize (all of them), unnormalized otherwise (the proteins)
    n_layer_channels = len(layer_keys)
    list_layer_numpydata = [functionPercNorm(np.single(channel)) if config.flag_normalize or is_layer_channel_normalized(key) else channel
                            for channel, key in zip(list_numpydata[:n_layer_channels], layer_keys)]

    list_segmentation, list_segmentation_match_nuclei, list_cell_props, mask_nuclei = \
        segment_layer_channels(list_layer_numpydata, layer_keys, dict_models, channel_labels)

    list_start_end, list_segmentation_filtered_layer = analyze_layers(list_layer_numpydata, list_segmentation, list_segmentation_match_nuclei,
                                                                      list_cell_props, mask_nuclei, layer_keys, channel_labels[:n_layer_channels],
                                                                      sample_name, folder_output_layers, config, dict_tables = dict_tables)

    # Positives channels (EdU - comment 8): each with its model on the percentile-normalized channel (whatever
    # flag_normalize is), outputs in the expression folder. Before the joint expression, which uses them.
    dict_segmentation_positives = {}  # {internal index: segmentation}
    for positives_number, positives_key in get_positives_channels(config.channel_names):
        if positives_number > n_channels:
            print(f'{sample_name}: {positives_key} channel {positives_number} but there are {n_channels} channels, '
                  f'no {positives_key} segmentation')
            continue
        i_positives = to_internal_index(positives_number, channel_order)
        segmentation_positives = segment_positives(list_numpydata[i_positives], dict_models[positives_key],
                                                   POSITIVES_MODEL_FILES[positives_key], channel_labels[i_positives],
                                                   sample_name, folder_output_expression)
        save_positives_histogram(segmentation_positives, list_cell_props[0], list_start_end[0], channel_labels[0],
                                 config.n_bins_edu, channel_labels[i_positives], sample_name, folder_output_expression,
                                 dict_tables = dict_tables)
        dict_segmentation_positives[i_positives] = segmentation_positives

    list_df_binned_expr = analyze_joint_expression(list_numpydata, list_segmentation_filtered_layer, list_start_end, channel_order,
                                                   channel_labels, sample_name, folder_output, config,
                                                   dict_segmentation_positives = dict_segmentation_positives,
                                                   dict_tables = dict_tables)

    # The per-sample csvs are already saved: the columns are added only for the aggregation across samples
    for df_table in list_df_binned_expr + list(dict_tables.values()):
        df_table.insert(0, 'treatment', treatment)
        df_table.insert(0, 'sample', sample_name)
    return list_df_binned_expr, dict_tables


def save_aggregated_tables(dict_list_tables, folder_aggregated):
    """Every per-sample csv with all the samples together: dict_list_tables is {file name without the sample name:
    [table of each sample, with 'sample'/'treatment' columns]}; each is saved in folder_aggregated as 'all_samples' +
    that file name (without the '_partial' of the Fig 3F/3K ones, which are partial only per sample)."""
    os.makedirs(folder_aggregated, exist_ok=True)
    for suffix, list_tables in dict_list_tables.items():
        # Empty tables (e.g. a sample without positive cells) only add the header, which the others already have
        list_non_empty = [df_table for df_table in list_tables if not df_table.empty]
        df_all = pd.concat(list_non_empty, ignore_index=True) if list_non_empty else list_tables[0]
        df_all.to_csv(os.path.join(folder_aggregated, 'all_samples' + suffix.replace('_partial', '')), index=False)
    print(f'{len(dict_list_tables)} csvs with all the samples together saved in {folder_aggregated}')


def run_figure_3(config):
    """Runs every sample in config and then the aggregation across samples, saved in the folder 'all_samples' +
    config.output_folder_ending inside config.folder_root: the binned expression (csv and boxplots per treatment, pair
    and channel) and every other per-sample csv (Fig 3 scatters, 3F, 3K, Fig 5 layers and edge fitting, EdU
    histogram), all the samples together in one csv each."""
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    validate_config(config)

    # The parameters of this run, in a txt in every output folder: this one, the aggregated csvs one and each
    # sample's folders (process_sample)
    folder_aggregated = os.path.join(config.folder_root, 'all_samples' + config.output_folder_ending)
    os.makedirs(folder_aggregated, exist_ok=True)
    parameters_text = get_parameters_text(config)
    save_parameters_txt(config.folder_root, parameters_text)
    save_parameters_txt(folder_aggregated, parameters_text)

    # One model per layer channel used and per positives channel used (EdU), loaded once and reused for every
    # sample; channels with the same model file (e.g. both mRNAs) share the loaded model
    keys_paths = [(key, get_layer_model_path(config.folder_models, key)) for key in get_used_layer_keys(config.channel_names)]
    keys_paths += [(key, get_positives_model_path(config.folder_models, key)) for _, key in get_positives_channels(config.channel_names)]
    dict_models = {}
    dict_models_by_path = {}
    for key, path_model in keys_paths:
        if path_model not in dict_models_by_path:
            print('Loading model ' + key + ': ' + path_model)
            dict_models_by_path[path_model] = models.CellposeModel(pretrained_model=path_model, gpu=config.flag_gpu)
        dict_models[key] = dict_models_by_path[path_model]

    list_df_binned_all = []  # binned expression of every sample, aggregated after the loop
    dict_list_tables = {}    # the other csvs: {file name without the sample name: [table of each sample]}
    for tiff_name, treatment in zip(config.tiffs_name, config.tiffs_treatment):
        list_df_binned_expr, dict_tables = process_sample(tiff_name, treatment, config, dict_models, parameters_text)
        list_df_binned_all.extend(list_df_binned_expr)
        for suffix, df_table in dict_tables.items():
            dict_list_tables.setdefault(suffix, []).append(df_table)
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    save_aggregated_tables(dict_list_tables, folder_aggregated)
    if list_df_binned_all:
        df_binned_all = pd.concat(list_df_binned_all, ignore_index=True)
        df_binned_all.to_csv(os.path.join(folder_aggregated, 'binned_expression_all_samples.csv'), index=False)
        binned_rows = [(name, n_bins_channel, get_binned_layers_name(config, layer_number_1, layer_number_2))
                       for number, name, n_bins_channel, layer_number_1, layer_number_2 in config.channels_binned_expression]
        plot_binned_expression_boxplots(df_binned_all, folder_aggregated, binned_rows)
    print('Done.')


def main():
    run_figure_3(build_config_from_parameters())


if __name__ == "__main__":
    main()

