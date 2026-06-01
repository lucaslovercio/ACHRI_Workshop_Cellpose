#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################ PARAMETERS ###################################################

folder_output = '/mnt/DATA/ACHRI/2024-10 Wilson Lab/2026-04 Publication/images_test_deliver6/output_tile_jpeg_20260527a'

flag_rebuild_csvs = False

#######################################################################################################

import os
import sys
import csv
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

folder_validation = os.path.join(folder_output, 'validation')

# ── Cell type definitions ─────────────────────────────────────────────────────

CELL_ORDER = ['m', 'o', 'l', 'e', 'n', 'p', 'r', 'j']
WBC_LABELS = {'m', 'o', 'l', 'e', 'n'}
CELL_NAMES = {
    'm': 'Macrophage',
    'o': 'Monocyte',
    'l': 'Lymphocyte',
    'e': 'Eosinophil',
    'n': 'Neutrophil',
    'p': 'Epithelial',
    'r': 'RBC',
    'j': 'Junk',
}

# ── CSV helpers ───────────────────────────────────────────────────────────────

def find_tile_corrected_csvs(folder_out):
    """
    Walks folder_out and returns list of (quadrant_name, tile_name, path)
    for every per-tile *_corrected_counts.csv file.
    Excludes quadrant-level and image-level summary files.
    """
    results = []
    for root, _dirs, files in os.walk(folder_out):
        for fname in files:
            if not fname.endswith('_corrected_counts.csv'):
                continue
            path      = os.path.join(root, fname)
            rel       = os.path.relpath(path, folder_out)
            parts     = rel.split(os.sep)
            # expected depth: quadrant_name / x{x}_y{y} / tile_name_corrected_counts.csv
            if len(parts) != 3:
                continue
            quadrant_name = parts[0]
            tile_name     = fname[:-len('_corrected_counts.csv')]
            results.append((quadrant_name, tile_name, path))
    results.sort()
    return results


def parse_tile_corrected_csv(path):
    """
    Parses a per-tile _corrected_counts.csv.
    Returns dict: label_str -> {'automatic': int, 'corrected': int}
    """
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = (row.get('Label') or '').strip()
            if lbl not in CELL_NAMES:
                continue
            try:
                data[lbl] = {
                    'automatic': int(row['Original']),
                    'corrected': int(row['Corrected']),
                }
            except (KeyError, ValueError):
                pass
    return data


def build_per_celltype_tables(tile_csvs):
    """
    Returns dict: label_str -> list of {'quadrant', 'tile', 'automatic', 'corrected'}
    """
    tables = defaultdict(list)
    for quadrant_name, tile_name, path in tile_csvs:
        cell_data = parse_tile_corrected_csv(path)
        for lbl, counts in cell_data.items():
            tables[lbl].append({
                'quadrant':  quadrant_name,
                'tile':      tile_name,
                'automatic': counts['automatic'],
                'corrected': counts['corrected'],
            })
    return tables


def write_per_celltype_csv(folder_val, lbl, rows):
    """Writes the per-cell-type comparison CSV."""
    name = CELL_NAMES[lbl]
    path = os.path.join(folder_val, f'validation_{lbl}_{name}.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Quadrant', 'Tile', 'Automatic', 'Corrected'])
        for r in rows:
            writer.writerow([r['quadrant'], r['tile'], r['automatic'], r['corrected']])
    return path


def read_per_celltype_csv(path):
    """Reads a per-cell-type comparison CSV (tolerates manual row deletions)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    'quadrant':  row.get('Quadrant', ''),
                    'tile':      row.get('Tile', ''),
                    'automatic': int(row['Automatic']),
                    'corrected': int(row['Corrected']),
                })
            except (KeyError, ValueError):
                pass
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def scatter_one(ax, auto, corr, title, show_xlabel=True, show_ylabel=True):
    """
    Draws scatter + linear fit + identity line on ax.
    Returns (slope, std_err, r_value, p_value) or Nones if not enough data.
    """
    n = len(auto)
    ax.set_title(title, fontsize=10, fontweight='bold')

    if show_xlabel:
        ax.set_xlabel('Automatic count', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Corrected count', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    if n == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='gray')
        return None, None, None, None

    ax.scatter(auto, corr, s=35, alpha=0.75, color='steelblue',
               edgecolors='white', linewidths=0.4, zorder=3)

    lim = max(auto.max(), corr.max()) * 1.08
    lim = max(lim, 1)
    ax.set_xlim([-lim * 0.04, lim])
    ax.set_ylim([-lim * 0.04, lim])

    # Identity line (y = x)
    ax.plot([0, lim], [0, lim], '--', color='#aaaaaa', linewidth=1, zorder=1, label='y = x')

    slope = std_err = r_value = p_value = None
    if n >= 3:
        res     = stats.linregress(auto, corr)
        slope   = res.slope
        std_err = res.stderr
        r_value = res.rvalue
        p_value = res.pvalue

        x_fit = np.array([0, lim])
        y_fit = slope * x_fit + res.intercept
        ax.plot(x_fit, y_fit, '-', color='#e74c3c', linewidth=1.5, zorder=2, label='Linear fit')

        stats_txt = (f'r = {r_value:.3f}\n'
                     f'slope = {slope:.3f} ± {std_err:.3f}\n'
                     f'p = {p_value:.2e}\n'
                     f'n = {n} tiles')
        ax.text(0.97, 0.04, stats_txt, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#dddddd', alpha=0.85))
    else:
        ax.text(0.97, 0.04, f'n = {n} (need ≥ 3 for fit)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='gray')

    return slope, std_err, r_value, p_value


def plot_individual(folder_val, lbl, rows):
    """Saves a standalone scatter plot for one cell type."""
    name = CELL_NAMES[lbl]
    auto = np.array([r['automatic'] for r in rows], dtype=float)
    corr = np.array([r['corrected'] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter_one(ax, auto, corr, f'{name}  [{lbl}]')
    plt.tight_layout()
    path = os.path.join(folder_val, f'validation_{lbl}_{name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_combined(folder_val, all_rows):
    """Saves a 2-row × 4-col overview figure with all cell types."""
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle('Automatic vs Corrected counts — per tile', fontsize=13, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for idx, lbl in enumerate(CELL_ORDER):
        ax   = fig.add_subplot(gs[idx // 4, idx % 4])
        rows = all_rows.get(lbl, [])
        auto = np.array([r['automatic'] for r in rows], dtype=float)
        corr = np.array([r['corrected'] for r in rows], dtype=float)
        show_x = (idx // 4 == 1)
        show_y = (idx % 4 == 0)
        scatter_one(ax, auto, corr, f'{CELL_NAMES[lbl]}  [{lbl}]',
                    show_xlabel=show_x, show_ylabel=show_y)

    path = os.path.join(folder_val, 'validation_all_celltypes.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(folder_output):
        print(f'Output folder not found: {folder_output}')
        sys.exit(1)

    os.makedirs(folder_validation, exist_ok=True)

    # ── Step 1: build or load per-cell-type comparison CSVs ──────────────────

    all_rows = {}

    for lbl in CELL_ORDER:
        name     = CELL_NAMES[lbl]
        csv_path = os.path.join(folder_validation, f'validation_{lbl}_{name}.csv')

        if not flag_rebuild_csvs and os.path.exists(csv_path):
            print(f'Reading existing CSV: {os.path.basename(csv_path)}')
            all_rows[lbl] = read_per_celltype_csv(csv_path)
        else:
            # First call: discover tile CSVs and build tables
            if not hasattr(main, '_tables'):
                print('Scanning for per-tile corrected CSVs...')
                tile_csvs = find_tile_corrected_csvs(folder_output)
                if not tile_csvs:
                    print('No *_corrected_counts.csv files found. '
                          'Run the GUI and save corrections first.')
                    sys.exit(1)
                print(f'  Found {len(tile_csvs)} tile CSV(s).')
                main._tables = build_per_celltype_tables(tile_csvs)

            rows = main._tables.get(lbl, [])
            if rows:
                write_per_celltype_csv(folder_validation, lbl, rows)
                print(f'Written: validation_{lbl}_{name}.csv  ({len(rows)} tiles)')
            all_rows[lbl] = rows

    # ── Step 2: individual plots ──────────────────────────────────────────────

    print('\nGenerating individual scatter plots...')
    for lbl in CELL_ORDER:
        rows = all_rows.get(lbl, [])
        if not rows:
            print(f'  {CELL_NAMES[lbl]}: no data, skipping.')
            continue
        path = plot_individual(folder_validation, lbl, rows)
        auto = np.array([r['automatic'] for r in rows])
        corr = np.array([r['corrected'] for r in rows])
        if len(rows) >= 3:
            r, p = stats.pearsonr(auto, corr)
            print(f'  {CELL_NAMES[lbl]:12s}  n={len(rows):3d}  r={r:.3f}  p={p:.2e}  → {os.path.basename(path)}')
        else:
            print(f'  {CELL_NAMES[lbl]:12s}  n={len(rows):3d}  (need ≥ 3 for statistics)  → {os.path.basename(path)}')

    # ── Step 3: combined overview ─────────────────────────────────────────────

    print('\nGenerating combined overview figure...')
    path = plot_combined(folder_validation, all_rows)
    print(f'  → {os.path.basename(path)}')

    print(f'\nDone. All validation outputs in:\n  {folder_validation}')


if __name__ == '__main__':
    main()
