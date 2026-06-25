import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

POINT_SIZE = 4
POINT_ALPHA = 0.6
VIOLIN_ALPHA = 0.5

def default_palette(treatments):
    colours = ['#4a90d9', '#e74c3c', '#27ae60', '#f39c12',
               '#8e44ad', '#16a085', '#d35400', '#2c3e50']
    return {t: colours[i % len(colours)] for i, t in enumerate(treatments)}

def plot_violin_pct(df_cell, treatment_col, col, ylabel, treatments):
    treat_colors = default_palette(treatments)
    """Violin + jittered points for a per-cell percentage column."""
    fig, ax = plt.subplots(figsize=(6, 4))

    positions = list(range(len(treatments)))
    data_list, valid_pos, valid_treat = [], [], []

    for pos, treatment in zip(positions, treatments):
        vals = df_cell.loc[df_cell[treatment_col] == treatment, col].dropna().values
        if len(vals) >= 2:
            data_list.append(vals)
            valid_pos.append(pos)
            valid_treat.append(treatment)

    if data_list:
        parts = ax.violinplot(data_list, positions=valid_pos, widths=0.6,
                              showmedians=True, showextrema=True, showmeans=False)
        for body, t in zip(parts['bodies'], valid_treat):
            body.set_facecolor(treat_colors[t])
            body.set_alpha(VIOLIN_ALPHA)
            body.set_edgecolor(treat_colors[t])
        for partname in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(1.2)

    for pos, treatment in zip(positions, treatments):
        vals = df_cell.loc[df_cell[treatment_col] == treatment, col].dropna().values
        if len(vals) == 0:
            continue
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(pos + jitter, vals,
                   s=POINT_SIZE ** 2, color=treat_colors[treatment],
                   alpha=POINT_ALPHA, zorder=3, edgecolors='none')

    tick_labels = [f'{t}\n(n={df_cell.loc[df_cell[treatment_col]==t, col].dropna().shape[0]})'
                   for t in treatments]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Treatment', fontsize=10)
    ax.set_title(ylabel, fontsize=10, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    