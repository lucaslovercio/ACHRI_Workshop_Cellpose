#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script_analyze_layers_gui.py

Graphical interface for script_analyze_layers.py (Figure 3 / Figure 5 analysis of multipage tiffs).

Tab "Images and channels":
  - Choose the folder with the images: every .tif/.tiff in it is listed, and next to each one the
    user writes its treatment (e.g. WT, KO). The first tiff must be multipage (one page per channel).
  - Preview of the (up to 4) channels of the first tiff, with a name box under each one. Channels named
    DAPI, SATB2, BCL11B, TBR1, MRNA-BCL11B or MRNA-TBR1 (any case) are layer channels, each segmented with its
    own model (the mRNAs as the proteins: layer delineation, edge fitting, joint expression).
    A channel named EdU (POSITIVES_MODEL_FILES) is segmented with its own model of positive cells (comment 8,
    outputs ending in _comment8), without layer delineation. Any other name has no segmentation.
  - Joint expression pairs: one row per pair of channels (any but DAPI) plotted against each other, X and Y
    ("+" adds a row). Each pair has all its outputs (scatter plots, Fig 3F ratio Y/X, Fig 3K halves, rgb
    images) and csvs.
  - Binned expression: one row per channel to bin ("+" adds a row), with its number of Y bins and the layer
    channel(s) whose layers give the range of the bins: the first one is required, the second one optional
    (empty: only the layer of the first one, e.g. the channel's own layer or DAPI's).
  - The models folder (with the Models_for_layer_delineation and Models_EdU subfolders); each model is
    found by name.
Tab "Optional parameters": the rest of the parameters, with the defaults of script_analyze_layers.py.

The analysis runs in the background; its output is shown in the log at the bottom.

The last models folder and the optional parameters changed from their defaults are saved in
script_analyze_layers_gui_settings.json (next to this .py) when the models folder is chosen, on Run and when
the window is closed, and loaded when the GUI starts.
"""

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)
import re
import json
import queue
import threading
import traceback

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if root_path not in sys.path:
    sys.path.append(root_path)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

import script_analyze_layers as fig3

MAX_CHANNELS = 4
PREVIEW_SIZE = 220
VALID_NAME = re.compile(r'^[A-Za-z0-9_-]+$')  # channel names end up in file names
# Last models folder and changed optional parameters, next to this .py
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'script_analyze_layers_gui_settings.json')


def to_preview_image(channel, size=PREVIEW_SIZE):
    """2D channel -> PIL image for display: 1-99.5 percentile stretch to uint8, fit in size x size."""
    channel = np.asarray(channel, dtype=np.float32)
    low, high = np.percentile(channel, [1, 99.5])
    scaled = np.clip((channel - low) / (high - low + 1e-7), 0, 1)
    img = Image.fromarray(np.uint8(scaled * 255))
    img.thumbnail((size, size))
    return img


class QueueWriter:
    """File-like object that sends everything written to it to a queue (stdout/stderr -> GUI log)."""
    def __init__(self, q):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(text)

    def flush(self):
        pass


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, height=200, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, height=height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')


class AnalyzeLayersGUI:
    def __init__(self, root):
        self.root = root
        root.title('Figure 3 - neuron layers and expression')

        self.folder_var = tk.StringVar()
        self.tiff_names = []
        self.treatment_vars = []
        self.n_channels_first_tiff = 0
        self.channel_name_vars = [tk.StringVar() for _ in range(MAX_CHANNELS)]
        self.channel_name_vars[0].set('DAPI')
        self.channel_name_entries = []
        self.preview_labels = []
        self.preview_photos = [None] * MAX_CHANNELS  # keep references, or Tk drops the images
        self.pair_rows = []  # [(frame, x_name_var, y_name_var)]
        self.binned_rows = []  # [(frame, name_var, n_bins_var, layer_1_var, layer_2_var)]
        self.models_folder_var = tk.StringVar()
        self.models_status_var = tk.StringVar(value='Choose the folder with the Models_for_layer_delineation and Models_EdU subfolders.')
        self.optional_vars = {}
        self.log_queue = queue.Queue()
        self.worker = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True, padx=8, pady=8)
        tab_main = ttk.Frame(notebook, padding=8)
        tab_optional = ttk.Frame(notebook, padding=8)
        notebook.add(tab_main, text='Images and channels')
        notebook.add(tab_optional, text='Optional parameters')

        self.build_main_tab(tab_main)
        self.build_optional_tab(tab_optional)
        self.build_run_area(root)
        self.load_settings()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.after(100, self.poll_log)

    # ── Layout ────────────────────────────────────────────────────────────────

    def build_main_tab(self, tab):
        # Folder
        frame_folder = ttk.Frame(tab)
        frame_folder.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(frame_folder, text='Images folder:').pack(side='left')
        ttk.Entry(frame_folder, textvariable=self.folder_var, width=80, state='readonly').pack(side='left', padx=4, fill='x', expand=True)
        ttk.Button(frame_folder, text='Open folder...', command=self.choose_folder).pack(side='left')

        # Samples + treatments
        frame_samples = ttk.LabelFrame(tab, text='Samples: write the treatment of each image', padding=4)
        frame_samples.grid(row=1, column=0, sticky='nsew', padx=(0, 8))
        self.samples_scroll = ScrollableFrame(frame_samples, height=420)
        self.samples_scroll.pack(fill='both', expand=True)

        # Right column: channels, roles, binned expression, models
        right = ttk.Frame(tab)
        right.grid(row=1, column=1, sticky='nsew')
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        frame_channels = ttk.LabelFrame(right, text='Channels (preview of the first image) - write the name of each channel used', padding=4)
        frame_channels.pack(fill='x')
        black = ImageTk.PhotoImage(Image.new('L', (PREVIEW_SIZE, PREVIEW_SIZE)))
        self.black_photo = black
        for i in range(MAX_CHANNELS):
            col = ttk.Frame(frame_channels)
            col.grid(row=0, column=i, padx=4)
            preview = ttk.Label(col, image=black)
            preview.pack()
            self.preview_labels.append(preview)
            row_name = ttk.Frame(col)
            row_name.pack(pady=2)
            ttk.Label(row_name, text=f'C{i + 1}').pack(side='left')
            entry = ttk.Entry(row_name, textvariable=self.channel_name_vars[i], width=14)
            entry.pack(side='left', padx=2)
            self.channel_name_entries.append(entry)
        ttk.Label(frame_channels, foreground='gray', wraplength=4 * PREVIEW_SIZE,
                  text=f'Layer channels, each segmented with its own model and with layer delineation: {", ".join(fig3.LAYER_MODEL_FILES)} '
                       '(any case; one channel must be DAPI). '
                       f'{", ".join(fig3.POSITIVES_MODEL_FILES)} (any case): segmented with its own model of positive cells, without layer delineation '
                       f'(comment 8: outputs ending in {fig3.ENDING_EDU_OUTPUTS}, in the expression folder). '
                       'Any other name has no segmentation: it is only quantified in the DAPI cells.'
                  ).grid(row=1, column=0, columnspan=MAX_CHANNELS, sticky='w', pady=(4, 0))

        frame_pairs = ttk.LabelFrame(right, text='Joint expression pairs (channels other than DAPI): scatter plots, Fig 3F ratio Y/X, Fig 3K', padding=4)
        frame_pairs.pack(fill='x', pady=8)
        header = ttk.Frame(frame_pairs)
        header.pack(fill='x')
        ttk.Label(header, text='X axis', width=16).pack(side='left')
        ttk.Label(header, text='Y axis', width=16).pack(side='left', padx=(8, 0))
        ttk.Button(header, text='+', width=3, command=self.add_pair_row).pack(side='right')
        self.pairs_container = ttk.Frame(frame_pairs)
        self.pairs_container.pack(fill='x')

        frame_binned = ttk.LabelFrame(right, text='Binned expression (mean/median per Y bin, between the layer limits of 1 or 2 layer '
                                                  'channels; boxplots per treatment)', padding=4)
        frame_binned.pack(fill='x')
        header = ttk.Frame(frame_binned)
        header.pack(fill='x')
        ttk.Label(header, text='Channel', width=16).pack(side='left')
        ttk.Label(header, text='N bins', width=8).pack(side='left')
        ttk.Label(header, text='Layers of', width=16).pack(side='left', padx=(8, 0))
        ttk.Label(header, text='and of (optional)', width=16).pack(side='left', padx=(8, 0))
        ttk.Button(header, text='+', width=3, command=self.add_binned_row).pack(side='right')
        self.binned_container = ttk.Frame(frame_binned)
        self.binned_container.pack(fill='x')

        frame_models = ttk.LabelFrame(right, text='Segmentation models (Cellpose)', padding=4)
        frame_models.pack(fill='x', pady=8)
        ttk.Label(frame_models, text='Models folder:').grid(row=0, column=0, sticky='w')
        ttk.Entry(frame_models, textvariable=self.models_folder_var, width=60, state='readonly').grid(row=0, column=1, sticky='ew', padx=4, pady=1)
        ttk.Button(frame_models, text='Open folder...', command=self.choose_models_folder).grid(row=0, column=2)
        ttk.Label(frame_models, textvariable=self.models_status_var, foreground='gray').grid(row=1, column=0, columnspan=3, sticky='w')
        frame_models.columnconfigure(1, weight=1)

    def build_optional_tab(self, tab):
        defaults = fig3.AnalyzeLayersConfig.__dataclass_fields__
        for row, (name, description) in enumerate(fig3.OPTIONAL_PARAMETERS):
            default = defaults[name].default
            value_type = defaults[name].type  # annotated type (e.g. float for ratio_bin_min, even if the default is -5)
            ttk.Label(tab, text=name).grid(row=row, column=0, sticky='w', pady=2)
            if value_type is bool:
                var = tk.BooleanVar(value=default)
                ttk.Checkbutton(tab, variable=var).grid(row=row, column=1, sticky='w', padx=6)
            else:
                var = tk.StringVar(value=str(default))
                ttk.Entry(tab, textvariable=var, width=24).grid(row=row, column=1, sticky='w', padx=6)
            ttk.Label(tab, text=description, foreground='gray').grid(row=row, column=2, sticky='w')
            self.optional_vars[name] = (var, value_type)
        ttk.Button(tab, text='Restore defaults', command=self.restore_optional_defaults).grid(
            row=len(fig3.OPTIONAL_PARAMETERS), column=0, sticky='w', pady=8)

    def build_run_area(self, root):
        frame = ttk.Frame(root, padding=(8, 0, 8, 8))
        frame.pack(fill='both', expand=True)
        self.run_button = ttk.Button(frame, text='Run', command=self.run)
        self.run_button.pack(anchor='w')
        self.log_text = tk.Text(frame, height=12, state='disabled', wrap='word')
        self.log_text.pack(fill='both', expand=True, pady=(4, 0))

    def make_channel_combobox(self, parent, var, width=14, exclude_dapi=False, only_layers=False, allow_empty=False):
        """Combobox with the channel names entered so far (refreshed when opened): without DAPI if exclude_dapi, only
        the layer channels (fig3.LAYER_MODEL_FILES, the ones with layer delineation) if only_layers, and with an empty
        first option (to leave it unset) if allow_empty."""
        def get_values():
            names = self.get_channel_names_list()
            if exclude_dapi:
                names = [n for n in names if fig3.get_layer_key(n) != 'DAPI']
            if only_layers:
                names = [n for n in names if fig3.get_layer_key(n)]
            return ([''] if allow_empty else []) + names
        combobox = ttk.Combobox(parent, textvariable=var, state='readonly', width=width)
        combobox.configure(postcommand=lambda c=combobox: c.configure(values=get_values()))
        return combobox

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def choose_folder(self):
        folder = filedialog.askdirectory(title='Folder with the multipage tiffs')
        if not folder:
            return
        tiff_names = fig3.list_tiffs_in_folder(folder)
        if not tiff_names:
            messagebox.showerror('No images', 'There are no .tif/.tiff files in\n' + folder)
            return
        first_path = os.path.join(folder, tiff_names[0])
        n_pages = fig3.get_number_of_pages(first_path)
        if n_pages < 2:
            messagebox.showerror('Not multipage', f'{tiff_names[0]} is not a multipage tiff (it has {n_pages} page).\n'
                                 'Each image must have one page per channel.')
            return
        if n_pages > MAX_CHANNELS:
            messagebox.showwarning('Too many channels', f'{tiff_names[0]} has {n_pages} pages; only the first {MAX_CHANNELS} '
                                   'are shown and can be named.')

        self.folder_var.set(folder)
        self.tiff_names = tiff_names
        self.n_channels_first_tiff = min(n_pages, MAX_CHANNELS)
        self.fill_samples_table()
        self.show_preview(first_path)

    def fill_samples_table(self):
        for child in self.samples_scroll.inner.winfo_children():
            child.destroy()
        self.treatment_vars = []
        ttk.Label(self.samples_scroll.inner, text='Image', font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(self.samples_scroll.inner, text='Treatment', font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=1, sticky='w')
        for row, name in enumerate(self.tiff_names, start=1):
            ttk.Label(self.samples_scroll.inner, text=name).grid(row=row, column=0, sticky='w', padx=(0, 8))
            var = tk.StringVar()
            ttk.Entry(self.samples_scroll.inner, textvariable=var, width=12).grid(row=row, column=1, sticky='w', pady=1)
            self.treatment_vars.append(var)

    def show_preview(self, tiff_path):
        try:
            channels = fig3.read_tiff_channels(tiff_path)
        except Exception as e:
            messagebox.showerror('Could not read image', f'{tiff_path}\n\n{e}')
            return
        for i in range(MAX_CHANNELS):
            if i < len(channels):
                photo = ImageTk.PhotoImage(to_preview_image(channels[i]))
                self.preview_photos[i] = photo
                self.preview_labels[i].configure(image=photo)
                self.channel_name_entries[i].configure(state='normal')
            else:  # e.g. 3-channel images: 4th preview stays black and can't be named
                self.preview_photos[i] = None
                self.preview_labels[i].configure(image=self.black_photo)
                self.channel_name_vars[i].set('')
                self.channel_name_entries[i].configure(state='disabled')

    def choose_models_folder(self):
        folder = filedialog.askdirectory(title='Folder with the Models_for_layer_delineation and Models_EdU subfolders',
                                         initialdir=self.models_folder_var.get() or None)
        if not folder:
            return
        self.set_models_folder(folder)
        self.save_settings()

    def set_models_folder(self, folder):
        """Sets the models folder and shows which models it has."""
        self.models_folder_var.set(folder)
        found =[key for key in fig3.LAYER_MODEL_FILES if os.path.exists(fig3.get_layer_model_path(folder, key))]
        missing = [key for key in fig3.LAYER_MODEL_FILES if key not in found]
        status = 'Layer models found: ' + (', '.join(found) if found else 'none')
        if missing:
            # each file once (both mRNAs share one)
            status += f'  |  missing in {fig3.SUBFOLDER_LAYER_MODELS}: ' + ', '.join(dict.fromkeys(fig3.get_layer_model_file(key) for key in missing))
        found = [key for key in fig3.POSITIVES_MODEL_FILES if os.path.exists(fig3.get_positives_model_path(folder, key))]
        missing = [key for key in fig3.POSITIVES_MODEL_FILES if key not in found]
        status += '  |  Positives models found: ' + (', '.join(found) if found else 'none')
        if missing:
            status += (f'  |  missing in {fig3.SUBFOLDER_EDU_MODEL} (needed only if a channel has that name): '
                       + ', '.join(f'{key} ({fig3.POSITIVES_MODEL_FILES[key]})' for key in missing))
        self.models_status_var.set(status)

    def add_pair_row(self):
        frame = ttk.Frame(self.pairs_container)
        frame.pack(fill='x', pady=1)
        x_name_var = tk.StringVar()
        y_name_var = tk.StringVar()
        self.make_channel_combobox(frame, x_name_var, exclude_dapi=True).pack(side='left')
        self.make_channel_combobox(frame, y_name_var, exclude_dapi=True).pack(side='left', padx=8)
        row = (frame, x_name_var, y_name_var)
        ttk.Button(frame, text='-', width=3, command=lambda: self.remove_row(self.pair_rows, row)).pack(side='left')
        self.pair_rows.append(row)

    def add_binned_row(self):
        frame = ttk.Frame(self.binned_container)
        frame.pack(fill='x', pady=1)
        name_var = tk.StringVar()
        n_bins_var = tk.StringVar(value='4')
        layer_1_var = tk.StringVar()
        layer_2_var = tk.StringVar()
        self.make_channel_combobox(frame, name_var).pack(side='left')
        ttk.Spinbox(frame, from_=1, to=100, textvariable=n_bins_var, width=6).pack(side='left', padx=8)
        self.make_channel_combobox(frame, layer_1_var, only_layers=True).pack(side='left', padx=(8, 0))
        self.make_channel_combobox(frame, layer_2_var, only_layers=True, allow_empty=True).pack(side='left', padx=8)
        row = (frame, name_var, n_bins_var, layer_1_var, layer_2_var)
        ttk.Button(frame, text='-', width=3, command=lambda: self.remove_row(self.binned_rows, row)).pack(side='left')
        self.binned_rows.append(row)

    def remove_row(self, rows, row):
        """Removes row (its frame is row[0]) from the panel and from rows (self.pair_rows or self.binned_rows)."""
        row[0].destroy()
        rows.remove(row)

    def restore_optional_defaults(self):
        defaults = fig3.AnalyzeLayersConfig.__dataclass_fields__
        for name, (var, _) in self.optional_vars.items():
            default = defaults[name].default
            var.set(default if isinstance(default, bool) else str(default))

    # ── Settings (SETTINGS_PATH) ──────────────────────────────────────────────

    def get_changed_optional_values(self):
        """{name: value as written in the GUI} of the optional parameters that differ from their defaults
        (so a default changed later in script_analyze_layers.py is used unless the user had changed it)."""
        defaults = fig3.AnalyzeLayersConfig.__dataclass_fields__
        changed = {}
        for name, (var, value_type) in self.optional_vars.items():
            default = defaults[name].default
            if value_type is bool:
                value, default = bool(var.get()), bool(default)
            else:
                value, default = var.get().strip(), str(default)
            if value != default:
                changed[name] = value
        return changed

    def save_settings(self):
        settings = {'models_folder': self.models_folder_var.get(),
                    'optional_parameters': self.get_changed_optional_values()}
        try:
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(settings, f, indent=2)
        except OSError as e:
            self.log(f'Could not save the settings in {SETTINGS_PATH}: {e}\n')

    def load_settings(self):
        if not os.path.exists(SETTINGS_PATH):
            return
        try:
            with open(SETTINGS_PATH) as f:
                settings = json.load(f)
            models_folder = settings.get('models_folder', '')
            optional_parameters = settings.get('optional_parameters', {})
        except (OSError, ValueError, AttributeError) as e:
            self.log(f'Could not read the settings in {SETTINGS_PATH} ({e}), using the defaults.\n')
            return

        if models_folder and os.path.isdir(models_folder):
            self.set_models_folder(models_folder)
        elif models_folder:
            self.models_status_var.set('Last models folder not found: ' + models_folder)

        for name, value in optional_parameters.items():
            if name in self.optional_vars:  # ignore parameters that don't exist anymore
                var, value_type = self.optional_vars[name]
                var.set(bool(value) if value_type is bool else str(value))
        changed = ', '.join(name for name in optional_parameters if name in self.optional_vars)
        self.log(f'Settings loaded from {SETTINGS_PATH}' + (f' (changed parameters: {changed})' if changed else '') + '\n')

    def on_close(self):
        self.save_settings()
        self.root.destroy()

    # ── Config ────────────────────────────────────────────────────────────────

    def get_channel_names_list(self):
        return [v.get().strip() for v in self.channel_name_vars[:self.n_channels_first_tiff] if v.get().strip()]

    def build_config(self):
        """AnalyzeLayersConfig from the GUI fields. Raises ValueError with a readable message if something is missing/wrong."""
        if not self.tiff_names:
            raise ValueError('Open the folder with the images first.')

        treatments = [v.get().strip() for v in self.treatment_vars]
        missing = [name for name, t in zip(self.tiff_names, treatments) if not t]
        if missing:
            raise ValueError('Write the treatment of every image. Missing:\n' + '\n'.join(missing))

        # Name -> viewer channel number (C1 = 1, ...)
        name_to_number = {}
        for i, var in enumerate(self.channel_name_vars[:self.n_channels_first_tiff]):
            name = var.get().strip()
            if not name:
                continue
            if not VALID_NAME.match(name):
                raise ValueError(f'Channel name "{name}" (C{i + 1}): use only letters, numbers, "-" and "_" (it is used in file names).')
            if name in name_to_number:
                raise ValueError(f'Channel name "{name}" is used twice.')
            name_to_number[name] = i + 1

        pairs_joint_expression = []
        for _, x_name_var, y_name_var in self.pair_rows:
            if x_name_var.get() not in name_to_number or y_name_var.get() not in name_to_number:
                raise ValueError('Choose the X and Y channels of every joint expression pair (or remove the row with "-").')
            pairs_joint_expression.append((name_to_number[x_name_var.get()], name_to_number[y_name_var.get()]))

        channels_binned_expression = []
        for _, name_var, n_bins_var, layer_1_var, layer_2_var in self.binned_rows:
            name = name_var.get()
            if name not in name_to_number:
                raise ValueError('Choose the channel of every binned expression row (or remove the row with "-").')
            try:
                n_bins = int(n_bins_var.get())
                assert n_bins >= 1
            except (ValueError, AssertionError):
                raise ValueError(f'Number of bins of {name} must be an integer >= 1.')
            if layer_1_var.get() not in name_to_number:
                raise ValueError(f'Binned expression of {name}: choose the layer channel whose layer gives the range of the bins '
                                 '("Layers of"; "and of" is optional).')
            # Second layer channel optional: empty (or a name that is not there anymore) -> None
            layer_number_2 = name_to_number.get(layer_2_var.get()) if layer_2_var.get() else None
            if layer_2_var.get() and layer_number_2 is None:
                raise ValueError(f'Binned expression of {name}: choose the second layer channel again (or leave it empty).')
            channels_binned_expression.append((name_to_number[name], name, n_bins, name_to_number[layer_1_var.get()], layer_number_2))

        if not self.models_folder_var.get():
            raise ValueError('Choose the models folder.')

        optional = {}
        for name, (var, value_type) in self.optional_vars.items():
            if value_type is bool:
                optional[name] = bool(var.get())
                continue
            try:
                optional[name] = value_type(var.get())
            except ValueError:
                raise ValueError(f'Optional parameter {name}: "{var.get()}" is not a valid {value_type.__name__}.')

        config = fig3.AnalyzeLayersConfig(
            folder_root=self.folder_var.get(), tiffs_name=list(self.tiff_names), tiffs_treatment=treatments,
            channel_names={number: name for name, number in name_to_number.items()},
            pairs_joint_expression=pairs_joint_expression,
            channels_binned_expression=channels_binned_expression,
            folder_models=self.models_folder_var.get(),
            **optional)
        # DAPI named, layer and positives channels used once, joint expression pairs, models present for the named channels
        try:
            fig3.validate_config(config)
        except FileNotFoundError as e:
            raise ValueError(str(e))
        return config

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        try:
            config = self.build_config()
        except ValueError as e:
            messagebox.showerror('Check the parameters', str(e))
            return
        self.save_settings()
        self.run_button.configure(state='disabled')
        self.log('Starting...\n')
        self.worker = threading.Thread(target=self.run_worker, args=(config,), daemon=True)
        self.worker.start()

    def run_worker(self, config):
        writer = QueueWriter(self.log_queue)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = writer, writer
        try:
            fig3.run_figure_3(config)
            self.log_queue.put(('DONE', None))
        except Exception:
            self.log_queue.put(traceback.format_exc())
            self.log_queue.put(('DONE', 'error'))
        finally:
            sys.stdout, sys.stderr = stdout, stderr

    def poll_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == 'DONE':
                    self.run_button.configure(state='normal')
                    if item[1] == 'error':
                        messagebox.showerror('Error', 'The analysis stopped with an error - see the log.')
                    else:
                        messagebox.showinfo('Done', 'Analysis finished. Results are in\n' + self.folder_var.get())
                else:
                    self.log(item)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_log)

    def log(self, text):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', text)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')


def main():
    root = tk.Tk()
    AnalyzeLayersGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

