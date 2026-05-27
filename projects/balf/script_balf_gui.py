#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################ PARAMETERS ###################################################

path_image    = ''
folder_output = ''

tile_size          = 2400
downsample_factor  = 4
jpeg_quality       = 90
cells_per_quadrant = 20

color_quadrant_lines = (255, 0,   0)
color_tile_analyzed  = (0,   200, 0)
line_thickness       = 5
min_pixels_matching = 200
confidence_threshold = 0.55
border_rectangle = 5
tuple_orange = (255, 165, 0)
tuple_green =(0, 255, 0)

flag_gpu = True

folder_models                = ''
filename_model_cells         = 'cell_reinhard_original_and_jpegs_model_nuclei_diam_50_ji_0.7175.002349'
filename_model_nucleus_lobes = 'nuclei_reinhard_original_and_jpeg_model_nuclei_diam_30_ji_0.7266.571736'
filename_normalizer_reinhard = 'normalizer_Reinhard.pkl'
filename_rf_celltype         = '20260507_RF_Cell_classification.pkl'


#######################################################################################################

import numpy as np


import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

import csv
import time
import datetime
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

path_model_cells          = os.path.join(folder_models, filename_model_cells)
path_model_nucleus_lobes  = os.path.join(folder_models, filename_model_nucleus_lobes)
path_rf_celltypes         = os.path.join(folder_models, filename_rf_celltype)
path_transformer_reinhard = os.path.join(folder_models, filename_normalizer_reinhard)

from functions_balf import process_image, label_to_name
from aux_functions.draw_roi_over_image import draw_roi_over_rgb

# ── GUI constants ─────────────────────────────────────────────────────────────

ALL_LABEL        = '__all__'
MANUAL_CELL_SIZE = 60    # default bbox side for manually added cells (pixels)

COLOR_REMOVED = (200,  30,  30)
COLOR_MANUAL  = ( 30,  80, 210)

WBC_LABELS = {'m', 'o', 'l', 'e', 'n'}
CELL_ORDER  = ['m', 'o', 'l', 'e', 'n', 'p', 'r', 'j']


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CellRecord:
    cell_id:    int
    label_str:  str
    confidence: float
    bbox:       Tuple[int, int, int, int]   # (x, y, w, h) in tile pixels
    is_removed: bool = False
    is_manual:  bool = False


@dataclass
class TileRecord:
    tile_name:     str
    quadrant_name: str
    img_png_path:  str
    cells:         List[CellRecord] = field(default_factory=list)


# ── Pipeline helpers (identical to deliver_2) ─────────────────────────────────

def _ordered_labels(present):
    """Returns labels in canonical display order; unknown labels appended sorted."""
    known   = [l for l in CELL_ORDER if l in present]
    unknown = sorted(l for l in present if l not in CELL_ORDER)
    return known + unknown


def get_sample_name(path):
    filename = os.path.split(path)[1]
    for ext in ('.tiff', '.tif', '.jpg', '.jpeg', '.png'):
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def find_center_of_mass_green(channel_g_small, background_threshold=200):
    channel_g    = channel_g_small.astype(np.float32)
    mask_tissue  = channel_g < background_threshold
    weight       = (255.0 - channel_g) * mask_tissue
    total_weight = np.sum(weight)
    if total_weight == 0:
        print('Warning: no tissue found, falling back to geometric center.')
        h, w = channel_g_small.shape
        return w // 2, h // 2
    y_coords, x_coords = np.mgrid[0:channel_g_small.shape[0], 0:channel_g_small.shape[1]]
    cx = int(np.sum(x_coords * weight) / total_weight)
    cy = int(np.sum(y_coords * weight) / total_weight)
    return cx, cy


def draw_quadrant_lines(img_rgb, cx, cy, color, thickness):
    h, w   = img_rgb.shape[:2]
    result = img_rgb.copy()
    cv2.line(result, (cx, 0),  (cx, h), color, thickness)
    cv2.line(result, (0,  cy), (w, cy), color, thickness)
    return result


def draw_tile_grid(img_rgb, quadrants, tile_size, downsample_factor, color, thickness):
    result = img_rgb.copy()
    ts     = tile_size // downsample_factor
    for (x_start, y_start, x_end, y_end) in quadrants.values():
        xs, ys = x_start // downsample_factor, y_start // downsample_factor
        xe, ye = x_end   // downsample_factor, y_end   // downsample_factor
        x = xs
        while x <= xe:
            cv2.line(result, (x, ys), (x, ye), color, thickness)
            x += ts
        y = ys
        while y <= ye:
            cv2.line(result, (xs, y), (xe, y), color, thickness)
            y += ts
    return result


def draw_analyzed_tile_border(img_preview, x, y, tile_size, downsample_factor, color, thickness):
    ts  = tile_size // downsample_factor
    x_p = x // downsample_factor
    y_p = y // downsample_factor
    cv2.rectangle(img_preview, (x_p, y_p), (x_p + ts, y_p + ts), color, thickness)


def save_tile_png(pil_img, x, y, tile_size, img_w, img_h, path_out):
    x_end = min(x + tile_size, img_w)
    y_end = min(y + tile_size, img_h)
    tile  = pil_img.crop((x, y, x_end, y_end))
    if tile.size != (tile_size, tile_size):
        tile_padded = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
        tile_padded.paste(tile, (0, 0))
        tile = tile_padded
    cv2.imwrite(path_out, cv2.cvtColor(np.array(tile), cv2.COLOR_RGB2BGR))
    return path_out


def extract_cell_bboxes(img_segmentation_cells, cells_id):
    """Returns {cell_id: (x, y, w, h)} tight bounding boxes from the segmentation mask."""
    bboxes = {}
    for cell_id in cells_id:
        rows, cols = np.where(img_segmentation_cells == cell_id)
        if len(rows) == 0:
            continue
        x = int(cols.min())
        y = int(rows.min())
        bboxes[cell_id] = (x, y, int(cols.max()) - x + 1, int(rows.max()) - y + 1)
    return bboxes


def save_tile_outputs(folder_tile, tile_name, img_rgb_original, img_rgb,
                      img_segmentation_cells, img_segmentation_nuclei,
                      n_cells, pred_str, pred_confidence, cells_id):
    os.makedirs(folder_tile, exist_ok=True)

    composed = draw_roi_over_rgb(img_rgb_original.copy(), img_segmentation_cells,
                                 color_rectangle=tuple_orange, border_rectangle=border_rectangle)
    cv2.imwrite(os.path.join(folder_tile, tile_name + f'_AllCells_{n_cells}.jpg'),
                cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

    for label_str in set(pred_str):
        indexes     = [i for i, v in enumerate(pred_str) if v == label_str]
        confidences = np.array([pred_confidence[i] for i in indexes])
        id_cells    = [cells_id[i] for i in indexes]
        high_conf   = confidences > confidence_threshold

        img_low_mask  = np.zeros_like(img_segmentation_cells)
        img_conf_mask = np.zeros_like(img_segmentation_cells)
        for id_cell in np.array(id_cells)[~high_conf]:
            img_low_mask[img_segmentation_cells == id_cell] = id_cell
        for id_cell in np.array(id_cells)[high_conf]:
            img_conf_mask[img_segmentation_cells == id_cell] = id_cell

        color_image = draw_roi_over_rgb(img_rgb_original.copy(), img_low_mask,
                                        color_rectangle=tuple_orange, border_rectangle=border_rectangle)
        color_image = draw_roi_over_rgb(color_image, img_conf_mask,
                                        color_rectangle=tuple_green,  border_rectangle=border_rectangle)
        n_total = len(indexes)
        n_conf  = int(np.sum(high_conf))
        cv2.imwrite(
            os.path.join(folder_tile,
                         f'{tile_name}_class_{label_str}_{label_to_name(label_str)}'
                         f'_count_{n_total}_confident_{n_conf}.jpg'),
            cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

    fig, _ = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True, layout='constrained')
    fig.suptitle(f'Validation - {tile_name}', fontsize=12)
    plt.subplot(2, 2, 1); plt.imshow(img_rgb_original);                          plt.gca().set_title('Original')
    plt.subplot(2, 2, 2); plt.imshow(img_rgb);                                   plt.gca().set_title('Normalized')
    plt.subplot(2, 2, 3); plt.imshow(img_segmentation_cells);                    plt.gca().set_title('Cell')
    plt.subplot(2, 2, 4); plt.imshow(img_segmentation_nuclei, cmap='gist_ncar'); plt.gca().set_title('Nuclei')
    plt.savefig(os.path.join(folder_tile, tile_name + '_features.png'), dpi=400)
    plt.close(fig)

    tile_counts = build_counts_from_predictions(pred_str, pred_confidence, cells_id)
    write_cell_counts_csv(os.path.join(folder_tile, tile_name + '_counts.csv'),
                          tile_counts, min_pixels_matching, confidence_threshold)


def build_counts_from_predictions(pred_str, pred_confidence, cells_id):
    counts = {}
    for label_str in set(pred_str):
        indexes     = [i for i, v in enumerate(pred_str) if v == label_str]
        confidences = np.array([pred_confidence[i] for i in indexes])
        high_conf   = confidences > confidence_threshold
        counts[label_str] = {
            'full_name'  : label_to_name(label_str),
            'n_total'    : len(indexes),
            'n_confident': int(np.sum(high_conf)),
        }
    return counts


def accumulate_counts(accumulator, tile_counts):
    for label_str, data in tile_counts.items():
        if label_str not in accumulator:
            accumulator[label_str] = {'full_name': data['full_name'], 'n_total': 0, 'n_confident': 0}
        accumulator[label_str]['n_total']     += data['n_total']
        accumulator[label_str]['n_confident'] += data['n_confident']


def write_summary_files(folder_out, name, counts_per_tile, quadrant_totals,
                        min_pixels_matching, confidence_threshold):
    txt  = f'Pixels matching: {min_pixels_matching}\n'
    txt += f'Confidence threshold: {confidence_threshold}\n'
    txt += '=' * 60 + '\n'
    for tile_name, counts in counts_per_tile:
        txt += f'\nTile: {tile_name}\n'
        txt += f'{"Label":<6} {"Cell Type":<15} {"Detected":>9} {"Confident":>10}\n'
        txt += '-' * 45 + '\n'
        tile_total = tile_conf = 0
        for label_str, data in counts.items():
            txt += f'{label_str:<6} {data["full_name"]:<15} {data["n_total"]:>9} {data["n_confident"]:>10}\n'
            tile_total += data['n_total']
            tile_conf  += data['n_confident']
        txt += f'{"Tile Total":<22} {tile_total:>9} {tile_conf:>10}\n'
    txt += '\n' + '=' * 60 + '\nQUADRANT TOTAL\n'
    txt += f'{"Label":<6} {"Cell Type":<15} {"Detected":>9} {"Confident":>10}\n'
    txt += '-' * 45 + '\n'
    grand_total = grand_conf = 0
    for label_str, data in quadrant_totals.items():
        txt += f'{label_str:<6} {data["full_name"]:<15} {data["n_total"]:>9} {data["n_confident"]:>10}\n'
        grand_total += data['n_total']
        grand_conf  += data['n_confident']
    txt += f'{"Grand Total":<22} {grand_total:>9} {grand_conf:>10}\n'
    with open(os.path.join(folder_out, name + '_summary.txt'), 'w') as f:
        f.write(txt)

    with open(os.path.join(folder_out, name + '_summary.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tile', 'Label', 'Cell Type', 'Detected', 'Confident', 'Manual Count'])
        for tile_name, counts in counts_per_tile:
            for label_str, data in counts.items():
                writer.writerow([tile_name, label_str, data['full_name'],
                                 data['n_total'], data['n_confident'], ''])
            tile_total = sum(d['n_total']     for d in counts.values())
            tile_conf  = sum(d['n_confident'] for d in counts.values())
            writer.writerow([tile_name, '', 'Tile Total', tile_total, tile_conf, ''])
            writer.writerow([])
        writer.writerow(['QUADRANT TOTAL', '', '', '', '', ''])
        for label_str, data in quadrant_totals.items():
            writer.writerow(['', label_str, data['full_name'],
                             data['n_total'], data['n_confident'], ''])
        writer.writerow(['', '', 'Grand Total', grand_total, grand_conf, ''])


def write_cell_counts_csv(path_csv, counts_dict, min_pixels_matching, confidence_threshold):
    ordered     = _ordered_labels(counts_dict.keys())
    wbc_present = [l for l in ordered if l in WBC_LABELS]
    wbc_total   = sum(counts_dict[l]['n_total']     for l in wbc_present)
    wbc_conf    = sum(counts_dict[l]['n_confident'] for l in wbc_present)
    grand_total = sum(d['n_total']     for d in counts_dict.values())
    grand_conf  = sum(d['n_confident'] for d in counts_dict.values())

    with open(path_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Pixels matching', min_pixels_matching])
        writer.writerow(['Confidence threshold', confidence_threshold])
        writer.writerow([])
        writer.writerow(['Label', 'Cell Type', 'Detected', 'Confident', '% of WBC', 'Manual Count'])
        for label_str in ordered:
            if label_str not in counts_dict:
                continue
            data = counts_dict[label_str]
            pct  = f'{data["n_total"] / wbc_total * 100:.1f}%' if label_str in WBC_LABELS and wbc_total > 0 else '—'
            writer.writerow([label_str, data['full_name'], data['n_total'], data['n_confident'], pct, ''])
        writer.writerow([])
        if wbc_present:
            pct_total = '100.0%' if wbc_total > 0 else '—'
            writer.writerow(['', 'WBC Total', wbc_total, wbc_conf, pct_total, ''])
        writer.writerow(['', 'Grand Total', grand_total, grand_conf, '', ''])


# ── Review GUI ────────────────────────────────────────────────────────────────

class ReviewGUI:
    def __init__(self, root: tk.Tk, tile_records: List[TileRecord],
                 sample_name: str, folder_output: str):
        self.root         = root
        self.tile_records = tile_records
        self.sample_name  = sample_name
        self.folder_out   = folder_output

        self.tile_idx        = 0
        self.current_label   = ALL_LABEL
        self._next_manual_id = -1

        self.session_start = time.time()
        self.tile_entry_t  = time.time()
        self.tile_times: Dict[str, float] = {}

        self.scale         = 1.0
        self.canvas_offset = (0, 0)
        self.img_display   = None
        self._photo        = None

        self.all_label_strs = sorted({c.label_str for rec in tile_records for c in rec.cells})

        self.root.title(f'Cell Count Review — {sample_name}')
        self.root.configure(bg='#1e1e1e')
        self._build_ui()
        self._load_tile(0)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._tick_timer()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        sidebar = tk.Frame(self.root, bg='#252526', width=240)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text=self.sample_name, bg='#252526', fg='#cccccc',
                 font=('Helvetica', 10, 'bold'), wraplength=220).pack(pady=(10, 2), padx=6)
        self._lbl_tile = tk.Label(sidebar, text='', bg='#252526', fg='#888888',
                                   font=('Helvetica', 8), wraplength=220)
        self._lbl_tile.pack(pady=(0, 8), padx=6)

        self._section(sidebar, 'Cell Type')
        cls_frame = tk.Frame(sidebar, bg='#252526')
        cls_frame.pack(fill=tk.X, padx=6)
        self._cls_btns: Dict[str, tk.Button] = {}
        for lbl in [ALL_LABEL] + self.all_label_strs:
            display = 'All Cells' if lbl == ALL_LABEL else label_to_name(lbl)
            btn = tk.Button(cls_frame, text=display, anchor='w',
                            bg='#3c3f41', fg='#cccccc', relief=tk.FLAT,
                            font=('Helvetica', 9), activebackground='#4a90d9',
                            command=lambda l=lbl: self._select_class(l))
            btn.pack(fill=tk.X, pady=1)
            self._cls_btns[lbl] = btn
        self._cls_btns[ALL_LABEL].config(bg='#4a90d9')

        self._section(sidebar, 'Counts (visible class)')
        cf = tk.Frame(sidebar, bg='#252526')
        cf.pack(fill=tk.X, padx=6)
        self._cnt_orig = self._count_row(cf, 'Original:',  '#cccccc')
        self._cnt_rem  = self._count_row(cf, 'Removed:',   '#ff7675')
        self._cnt_add  = self._count_row(cf, 'Added:',     '#74b9ff')
        self._cnt_corr = self._count_row(cf, 'Corrected:', '#55efc4')

        self._section(sidebar, 'Legend')
        lf = tk.Frame(sidebar, bg='#252526')
        lf.pack(fill=tk.X, padx=10)
        for color_hex, text in [
            ('#00c800', 'High confidence'),
            ('#ffa500', 'Low confidence'),
            ('#c81e1e', 'Removed  (click to restore)'),
            ('#1e50d2', 'Manually added  (click to delete)'),
        ]:
            row = tk.Frame(lf, bg='#252526')
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, bg=color_hex, width=3).pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=text, bg='#252526', fg='#aaaaaa',
                     font=('Helvetica', 8)).pack(side=tk.LEFT)

        self._section(sidebar, 'Review Time')
        self._lbl_timer = tk.Label(sidebar, text='00:00:00', bg='#252526', fg='#fdcb6e',
                                    font=('Courier', 16, 'bold'))
        self._lbl_timer.pack(pady=(0, 4))

        self._section(sidebar, 'Tiles')
        list_frame = tk.Frame(sidebar, bg='#252526')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6)
        sb = tk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tile_lb = tk.Listbox(list_frame, yscrollcommand=sb.set,
                                    bg='#2d2d2d', fg='#cccccc',
                                    selectbackground='#4a90d9',
                                    font=('Courier', 8), activestyle='none',
                                    borderwidth=0, highlightthickness=0)
        self._tile_lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._tile_lb.yview)
        for rec in self.tile_records:
            self._tile_lb.insert(tk.END, rec.tile_name.replace(self.sample_name + '_', ''))
        self._tile_lb.bind('<<ListboxSelect>>', self._on_listbox_select)

        tk.Button(sidebar, text='Save & Close', bg='#27ae60', fg='white',
                  font=('Helvetica', 10, 'bold'), relief=tk.FLAT, cursor='hand2',
                  command=self._save_and_close).pack(fill=tk.X, padx=6, pady=8)

        right = tk.Frame(self.root, bg='#1e1e1e')
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        tk.Label(right,
                 text='Left-click box: remove/restore   |   Right-click box: force remove'
                      '   |   Left-click empty: add cell',
                 bg='#1e1e1e', fg='#555555', font=('Helvetica', 8)).pack(side=tk.TOP)

        self.canvas = tk.Canvas(right, bg='#111111', cursor='crosshair', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self._on_left_click)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', lambda _e: self._render())

        nav = tk.Frame(right, bg='#252526')
        nav.pack(side=tk.BOTTOM, fill=tk.X)
        for text, cmd in [('◀ Prev', self._prev_tile), ('Next ▶', self._next_tile)]:
            tk.Button(nav, text=text, bg='#3c3f41', fg='#cccccc', relief=tk.FLAT,
                      font=('Helvetica', 9), cursor='hand2',
                      command=cmd).pack(side=tk.LEFT, padx=4, pady=4)
        self._lbl_nav = tk.Label(nav, text='', bg='#252526', fg='#888888', font=('Helvetica', 8))
        self._lbl_nav.pack(side=tk.LEFT, padx=8)

    def _section(self, parent, title):
        tk.Label(parent, text=title.upper(), bg='#252526', fg='#555555',
                 font=('Helvetica', 7, 'bold')).pack(anchor='w', padx=8, pady=(8, 2))

    def _count_row(self, parent, label, color):
        f = tk.Frame(parent, bg='#252526')
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, bg='#252526', fg='#888888',
                 font=('Helvetica', 9), width=10, anchor='w').pack(side=tk.LEFT)
        lbl = tk.Label(f, text='0', bg='#252526', fg=color,
                       font=('Helvetica', 10, 'bold'), anchor='e')
        lbl.pack(side=tk.RIGHT)
        return lbl

    # ── Navigation ───────────────────────────────────────────────────────────

    def _load_tile(self, idx):
        if not (0 <= idx < len(self.tile_records)):
            return
        prev_name = self.tile_records[self.tile_idx].tile_name
        self.tile_times[prev_name] = (self.tile_times.get(prev_name, 0.0)
                                      + time.time() - self.tile_entry_t)
        self.tile_idx    = idx
        self.tile_entry_t = time.time()

        rec = self.tile_records[idx]
        self._lbl_tile.config(text=rec.tile_name.replace(self.sample_name + '_', ''))
        self._lbl_nav.config(text=f'Tile {idx + 1} / {len(self.tile_records)}')

        img_bgr = cv2.imread(rec.img_png_path)
        self.img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None

        self._tile_lb.selection_clear(0, tk.END)
        self._tile_lb.selection_set(idx)
        self._tile_lb.see(idx)
        self._render()
        self._update_counts()

    def _prev_tile(self):  self._load_tile(self.tile_idx - 1)
    def _next_tile(self):  self._load_tile(self.tile_idx + 1)

    def _on_listbox_select(self, _event):
        sel = self._tile_lb.curselection()
        if sel:
            self._load_tile(sel[0])

    def _select_class(self, label):
        self.current_label = label
        for lbl, btn in self._cls_btns.items():
            btn.config(bg='#4a90d9' if lbl == label else '#3c3f41')
        self._render()
        self._update_counts()

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self):
        if self.img_display is None:
            return
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = self.img_display.shape[:2]
        self.scale = min(cw / w, ch / h)
        dw, dh = int(w * self.scale), int(h * self.scale)

        img = self.img_display.copy()
        for cell in self.tile_records[self.tile_idx].cells:
            if self.current_label != ALL_LABEL and cell.label_str != self.current_label:
                continue
            x, y, bw, bh = cell.bbox
            x2, y2 = x + bw, y + bh
            if cell.is_removed:
                cv2.rectangle(img, (x, y), (x2, y2), COLOR_REMOVED, border_rectangle)
                cv2.line(img, (x, y), (x2, y2), COLOR_REMOVED, 1)
                cv2.line(img, (x2, y), (x, y2), COLOR_REMOVED, 1)
            elif cell.is_manual:
                cv2.rectangle(img, (x, y), (x2, y2), COLOR_MANUAL, border_rectangle)
            elif cell.confidence > confidence_threshold:
                cv2.rectangle(img, (x, y), (x2, y2), tuple_green, border_rectangle)
            else:
                cv2.rectangle(img, (x, y), (x2, y2), tuple_orange, border_rectangle)

        img_s = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img_s))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.canvas_offset = (ox, oy)
        self.canvas.delete('all')
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

    # ── Click handling ────────────────────────────────────────────────────────

    def _canvas_to_img(self, cx, cy):
        ox, oy = self.canvas_offset
        return int((cx - ox) / self.scale), int((cy - oy) / self.scale)

    def _cell_at(self, ix, iy) -> Optional[CellRecord]:
        for cell in self.tile_records[self.tile_idx].cells:
            if self.current_label != ALL_LABEL and cell.label_str != self.current_label:
                continue
            x, y, bw, bh = cell.bbox
            if x <= ix <= x + bw and y <= iy <= y + bh:
                return cell
        return None

    def _on_left_click(self, event):
        if self.img_display is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        h, w = self.img_display.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return
        cell = self._cell_at(ix, iy)
        if cell is not None:
            if cell.is_manual:
                self.tile_records[self.tile_idx].cells.remove(cell)
            else:
                cell.is_removed = not cell.is_removed
        else:
            self._add_cell(ix, iy)
        self._render()
        self._update_counts()

    def _on_right_click(self, event):
        if self.img_display is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        cell = self._cell_at(ix, iy)
        if cell is not None and not cell.is_manual:
            cell.is_removed = True
            self._render()
            self._update_counts()

    def _add_cell(self, ix, iy):
        lbl = self.current_label
        if lbl == ALL_LABEL:
            lbl = self._ask_class()
            if lbl is None:
                return
        half = MANUAL_CELL_SIZE // 2
        h, w = self.img_display.shape[:2]
        bx = max(0, min(ix - half, w - MANUAL_CELL_SIZE))
        by = max(0, min(iy - half, h - MANUAL_CELL_SIZE))
        self.tile_records[self.tile_idx].cells.append(
            CellRecord(cell_id=self._next_manual_id, label_str=lbl, confidence=1.0,
                       bbox=(bx, by, MANUAL_CELL_SIZE, MANUAL_CELL_SIZE), is_manual=True))
        self._next_manual_id -= 1

    def _ask_class(self) -> Optional[str]:
        dlg = tk.Toplevel(self.root)
        dlg.title('Add Cell — Select Type')
        dlg.configure(bg='#252526')
        dlg.resizable(False, False)
        dlg.grab_set()
        tk.Label(dlg, text='Select cell type:', bg='#252526', fg='#cccccc',
                 font=('Helvetica', 10)).pack(padx=16, pady=10)

        # Always show all defined cell types, not just those detected in the tiles
        options = CELL_ORDER
        var = tk.StringVar(value=options[0])
        for lbl in options:
            tk.Radiobutton(dlg, text=f'{label_to_name(lbl)}  [{lbl}]', variable=var, value=lbl,
                           bg='#252526', fg='#cccccc', selectcolor='#4a90d9',
                           activebackground='#252526',
                           font=('Helvetica', 9)).pack(anchor='w', padx=20, pady=1)

        result = [None]
        def ok():
            result[0] = var.get() or None
            dlg.destroy()
        bf = tk.Frame(dlg, bg='#252526')
        bf.pack(pady=8)
        tk.Button(bf, text='OK',     command=ok,           bg='#3c3f41', fg='white',
                  relief=tk.FLAT, width=8).pack(side=tk.LEFT,  padx=4)
        tk.Button(bf, text='Cancel', command=dlg.destroy,  bg='#3c3f41', fg='white',
                  relief=tk.FLAT, width=8).pack(side=tk.RIGHT, padx=4)
        dlg.wait_window()
        return result[0]

    # ── Counts display ────────────────────────────────────────────────────────

    def _update_counts(self):
        visible = [c for c in self.tile_records[self.tile_idx].cells
                   if self.current_label == ALL_LABEL or c.label_str == self.current_label]
        original  = sum(1 for c in visible if not c.is_manual)
        removed   = sum(1 for c in visible if not c.is_manual and c.is_removed)
        added     = sum(1 for c in visible if c.is_manual)
        self._cnt_orig.config(text=str(original))
        self._cnt_rem.config( text=str(removed))
        self._cnt_add.config( text=str(added))
        self._cnt_corr.config(text=str(original - removed + added))

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _tick_timer(self):
        e = int(time.time() - self.session_start)
        self._lbl_timer.config(text=f'{e // 3600:02d}:{(e % 3600) // 60:02d}:{e % 60:02d}')
        self.root.after(1000, self._tick_timer)

    # ── Save / Close ──────────────────────────────────────────────────────────

    def _flush_tile_time(self):
        name = self.tile_records[self.tile_idx].tile_name
        self.tile_times[name] = self.tile_times.get(name, 0.0) + time.time() - self.tile_entry_t
        self.tile_entry_t = time.time()

    def _on_close(self):
        if messagebox.askyesno('Exit', 'Save corrections before closing?', parent=self.root):
            self._do_save()
        self.root.destroy()

    def _save_and_close(self):
        self._do_save()
        self.root.destroy()

    def _do_save(self):
        self._flush_tile_time()
        total_time = time.time() - self.session_start
        self._write_outputs(total_time)
        messagebox.showinfo('Saved', f'Corrected results saved to:\n{self.folder_out}',
                            parent=self.root)

    # ── Output writing ────────────────────────────────────────────────────────

    @staticmethod
    def _cell_label_summary(cells: List[CellRecord]) -> Dict[str, dict]:
        data: Dict[str, dict] = defaultdict(
            lambda: {'full_name': '', 'n_original': 0, 'n_removed': 0, 'n_added': 0})
        for cell in cells:
            d = data[cell.label_str]
            d['full_name'] = label_to_name(cell.label_str)
            if cell.is_manual:
                d['n_added'] += 1
            else:
                d['n_original'] += 1
                if cell.is_removed:
                    d['n_removed'] += 1
        return data

    @staticmethod
    def _write_corrected_csv(path: str, data: Dict[str, dict]):
        ordered     = _ordered_labels(data.keys())
        wbc_present = [l for l in ordered if l in WBC_LABELS]
        wbc_orig    = sum(data[l]['n_original'] for l in wbc_present)
        wbc_rem     = sum(data[l]['n_removed']  for l in wbc_present)
        wbc_add     = sum(data[l]['n_added']    for l in wbc_present)
        wbc_corr    = wbc_orig - wbc_rem + wbc_add
        grand_orig  = sum(d['n_original'] for d in data.values())
        grand_rem   = sum(d['n_removed']  for d in data.values())
        grand_add   = sum(d['n_added']    for d in data.values())
        grand_corr  = grand_orig - grand_rem + grand_add

        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Label', 'Cell Type', 'Original', 'Removed', 'Added', 'Corrected', '% of WBC'])
            for lbl in ordered:
                if lbl not in data:
                    continue
                d    = data[lbl]
                corr = d['n_original'] - d['n_removed'] + d['n_added']
                pct  = f'{corr / wbc_corr * 100:.1f}%' if lbl in WBC_LABELS and wbc_corr > 0 else '—'
                w.writerow([lbl, d['full_name'], d['n_original'], d['n_removed'], d['n_added'], corr, pct])
            w.writerow([])
            if wbc_present:
                pct_total = '100.0%' if wbc_corr > 0 else '—'
                w.writerow(['', 'WBC Total', wbc_orig, wbc_rem, wbc_add, wbc_corr, pct_total])
            w.writerow(['', 'Grand Total', grand_orig, grand_rem, grand_add, grand_corr, ''])

    def _save_corrected_images(self):
        """Saves AllCells and per-class corrected JPGs based on the current CellRecord state."""
        for rec in self.tile_records:
            tile_dir = os.path.dirname(rec.img_png_path)
            img_bgr  = cv2.imread(rec.img_png_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            active = [c for c in rec.cells if not c.is_removed]

            # AllCells corrected
            img_all = img_rgb.copy()
            for cell in active:
                x, y, bw, bh = cell.bbox
                if cell.is_manual:
                    color = COLOR_MANUAL
                elif cell.confidence > confidence_threshold:
                    color = tuple_green
                else:
                    color = tuple_orange
                cv2.rectangle(img_all, (x, y), (x + bw, y + bh), color, border_rectangle)
            cv2.imwrite(
                os.path.join(tile_dir, f'{rec.tile_name}_AllCells_corrected_{len(active)}.jpg'),
                cv2.cvtColor(img_all, cv2.COLOR_RGB2BGR))

            # Per-class corrected
            for label_str in _ordered_labels({c.label_str for c in active}):
                class_cells = [c for c in active if c.label_str == label_str]
                img_cls = img_rgb.copy()
                for cell in class_cells:
                    x, y, bw, bh = cell.bbox
                    if cell.is_manual:
                        color = COLOR_MANUAL
                    elif cell.confidence > confidence_threshold:
                        color = tuple_green
                    else:
                        color = tuple_orange
                    cv2.rectangle(img_cls, (x, y), (x + bw, y + bh), color, border_rectangle)
                full_name = label_to_name(label_str)
                cv2.imwrite(
                    os.path.join(tile_dir,
                                 f'{rec.tile_name}_class_{label_str}_{full_name}'
                                 f'_corrected_{len(class_cells)}.jpg'),
                    cv2.cvtColor(img_cls, cv2.COLOR_RGB2BGR))

    def _write_outputs(self, total_time: float):
        self._save_corrected_images()
        by_quadrant: Dict[str, List[TileRecord]] = defaultdict(list)
        for rec in self.tile_records:
            by_quadrant[rec.quadrant_name].append(rec)

        img_agg: Dict[str, dict] = defaultdict(
            lambda: {'full_name': '', 'n_original': 0, 'n_removed': 0, 'n_added': 0})

        # Per-tile
        for rec in self.tile_records:
            tile_data = self._cell_label_summary(rec.cells)
            self._write_corrected_csv(
                os.path.join(os.path.dirname(rec.img_png_path),
                             rec.tile_name + '_corrected_counts.csv'),
                tile_data)
            for lbl, d in tile_data.items():
                g = img_agg[lbl]
                g['full_name']   = d['full_name']
                g['n_original'] += d['n_original']
                g['n_removed']  += d['n_removed']
                g['n_added']    += d['n_added']

        # Per-quadrant
        for qname, recs in by_quadrant.items():
            all_cells = [c for rec in recs for c in rec.cells]
            self._write_corrected_csv(
                os.path.join(self.folder_out, qname,
                             f'{self.sample_name}_{qname}_corrected_summary.csv'),
                self._cell_label_summary(all_cells))

        # Image-level
        self._write_corrected_csv(
            os.path.join(self.folder_out, f'{self.sample_name}_image_corrected_summary.csv'),
            img_agg)

        # Timing
        with open(os.path.join(self.folder_out, f'{self.sample_name}_review_timing.csv'),
                  'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Tile', 'Time (s)'])
            for rec in self.tile_records:
                w.writerow([rec.tile_name, f'{self.tile_times.get(rec.tile_name, 0.0):.1f}'])
            w.writerow(['TOTAL', f'{total_time:.1f}'])
            w.writerow([])
            w.writerow(['Session Date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(path_image):
        print('Image not found. Stopping.')
        sys.exit(1)

    sample_name = get_sample_name(path_image)
    os.makedirs(folder_output, exist_ok=True)

    print('Reading image...')
    Image.MAX_IMAGE_PIXELS = None
    pil_img      = Image.open(path_image).convert('RGB')
    img_w, img_h = pil_img.size
    print(f'Image size: {img_w} x {img_h}')

    print('Extracting green channel...')
    channel_g_full  = np.array(pil_img.getchannel('G'))
    w_small         = img_w // downsample_factor
    h_small         = img_h // downsample_factor
    channel_g_small = cv2.resize(channel_g_full, (w_small, h_small), interpolation=cv2.INTER_AREA)
    del channel_g_full

    print('Finding center of mass...')
    cx_small, cy_small = find_center_of_mass_green(channel_g_small)
    del channel_g_small
    cx = cx_small * downsample_factor
    cy = cy_small * downsample_factor
    print(f'Center of mass: ({cx}, {cy})')

    quadrants = {
        'Q1_top_left'     : (0,   0,   cx,    cy),
        'Q2_top_right'    : (cx,  0,   img_w, cy),
        'Q3_bottom_left'  : (0,   cy,  cx,    img_h),
        'Q4_bottom_right' : (cx,  cy,  img_w, img_h),
    }

    print('Generating preview images...')
    pil_small = Image.open(path_image).convert('RGB').resize((w_small, h_small), Image.LANCZOS)
    img_small = np.array(pil_small)
    del pil_small

    img_preview_com = draw_quadrant_lines(img_small, cx_small, cy_small,
                                           color_quadrant_lines, line_thickness)
    cv2.imwrite(os.path.join(folder_output, sample_name + '_preview_center_of_mass.jpg'),
                cv2.cvtColor(img_preview_com, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    img_preview_grid = draw_tile_grid(img_preview_com, quadrants, tile_size,
                                       downsample_factor, (0, 0, 255), line_thickness)
    cv2.imwrite(os.path.join(folder_output, sample_name + '_preview_tile_grid.jpg'),
                cv2.cvtColor(img_preview_grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    img_tiles_analyzed = img_preview_grid.copy()
    del img_small, img_preview_com, img_preview_grid

    all_tile_records: List[TileRecord] = []
    image_totals = {}

    for quadrant_name, (x_start, y_start, x_end, y_end) in quadrants.items():
        print(f'\n{"="*60}\nProcessing {quadrant_name}\n{"="*60}')
        folder_quadrant  = os.path.join(folder_output, quadrant_name)
        os.makedirs(folder_quadrant, exist_ok=True)
        quadrant_totals  = {}
        counts_per_tile  = []
        n_cells_quadrant = 0

        y = y_start
        done = False
        while y < y_end and not done:
            x = x_start
            while x < x_end and not done:
                tile_name   = f'{sample_name}_{quadrant_name}_x{x}_y{y}'
                folder_tile = os.path.join(folder_quadrant, f'x{x}_y{y}')
                path_tile_png = os.path.join(folder_tile, tile_name + '.png')
                os.makedirs(folder_tile, exist_ok=True)
                print(f'\n  Tile: {tile_name}')

                save_tile_png(pil_img, x, y, tile_size, img_w, img_h, path_tile_png)

                img_rgb, img_rgb_original, img_segmentation_cells, img_segmentation_nuclei, \
                    n_cells, pred_str, pred_confidence, cells_id = \
                    process_image(path_tile_png,
                                  path_model_cells, path_model_nucleus_lobes,
                                  path_rf_celltypes, path_transformer_reinhard,
                                  min_pixels_matching = min_pixels_matching,
                                  flag_gpu=flag_gpu)
                print(f'  Cells found: {n_cells}')

                save_tile_outputs(folder_tile, tile_name, img_rgb_original, img_rgb,
                                  img_segmentation_cells, img_segmentation_nuclei,
                                  n_cells, pred_str, pred_confidence, cells_id)

                # Build TileRecord for the GUI
                bboxes = extract_cell_bboxes(img_segmentation_cells, cells_id)
                cell_records = [
                    CellRecord(cell_id=cells_id[i], label_str=pred_str[i],
                               confidence=pred_confidence[i], bbox=bboxes[cells_id[i]])
                    for i in range(len(cells_id)) if cells_id[i] in bboxes
                ]
                all_tile_records.append(TileRecord(
                    tile_name=tile_name, quadrant_name=quadrant_name,
                    img_png_path=path_tile_png, cells=cell_records))

                tile_counts = build_counts_from_predictions(pred_str, pred_confidence, cells_id)
                counts_per_tile.append((tile_name, tile_counts))
                accumulate_counts(quadrant_totals, tile_counts)
                accumulate_counts(image_totals,    tile_counts)

                n_cells_quadrant += n_cells
                print(f'  Quadrant total so far: {n_cells_quadrant}')

                draw_analyzed_tile_border(img_tiles_analyzed, x, y, tile_size,
                                          downsample_factor, color_tile_analyzed, line_thickness + 1)

                if n_cells_quadrant >= cells_per_quadrant:
                    print(f'  Reached {n_cells_quadrant} cells — stopping quadrant.')
                    done = True
                x += tile_size
            y += tile_size

        write_summary_files(folder_quadrant, f'{sample_name}_{quadrant_name}',
                            counts_per_tile, quadrant_totals,
                            min_pixels_matching, confidence_threshold)
        print(f'\n  {quadrant_name} done. Total cells: {n_cells_quadrant}')

    # Image-level original summary
    print(f'\n{"="*60}\nSaving image-level summary...')
    grand_total = sum(d['n_total']     for d in image_totals.values())
    grand_conf  = sum(d['n_confident'] for d in image_totals.values())
    txt  = f'Image: {sample_name}\nPixels matching: {min_pixels_matching}\n'
    txt += f'Confidence threshold: {confidence_threshold}\n' + '=' * 60 + '\n'
    txt += f'{"Label":<6} {"Cell Type":<15} {"Detected":>9} {"Confident":>10}\n' + '-' * 45 + '\n'
    for label_str, data in image_totals.items():
        txt += f'{label_str:<6} {data["full_name"]:<15} {data["n_total"]:>9} {data["n_confident"]:>10}\n'
    txt += f'{"Grand Total":<22} {grand_total:>9} {grand_conf:>10}\n'
    with open(os.path.join(folder_output, sample_name + '_image_summary.txt'), 'w') as f:
        f.write(txt)
    write_cell_counts_csv(os.path.join(folder_output, sample_name + '_image_summary.csv'),
                          image_totals, min_pixels_matching, confidence_threshold)

    cv2.imwrite(os.path.join(folder_output, sample_name + '_tiles_analyzed.jpg'),
                cv2.cvtColor(img_tiles_analyzed, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Launch correction GUI
    print('\nLaunching review GUI...')
    root = tk.Tk()
    root.geometry('1400x900')
    ReviewGUI(root, all_tile_records, sample_name, folder_output)
    root.mainloop()
    print(f'\nDone. All outputs in: {folder_output}')


if __name__ == '__main__':
    main()
