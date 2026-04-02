#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_ind
import cv2

def draw_concentric_circles( rgb_colors, radius=30, img_size=None, background=(0, 0, 0)
):
    """
    Draw 3 filled concentric circles with given RGB colors.

    Parameters
    ----------
    rgb_colors : list or tuple of 3 RGB tuples
        [(R,G,B), (R,G,B), (R,G,B)]
    radius : int
        Radius of the outermost circle
    img_size : int or None
        Size of output square image; defaults to 2*radius + 2
    background : RGB tuple
        Background color

    Returns
    -------
    img : np.ndarray
        OpenCV image (uint8, BGR)
    """

    assert len(rgb_colors) == 3, "Provide exactly 3 RGB colors"

    # Convert colors to 0–255 if needed and to BGR
    def to_bgr(c):
        if max(c) <= 1.0:
            c = [int(255 * x) for x in c]
        return (c[2], c[1], c[0])

    bgr_colors = []
    for c in rgb_colors:
        bgr_colors.append(to_bgr(c))
    bg = to_bgr(background)

    if img_size is None:
        img_size = 2 * radius + 2

    img = np.full((img_size, img_size, 3), bg, dtype=np.uint8)
    center = (img_size // 2, img_size // 2)

    # Radii for outer → inner
    radii = np.linspace(radius, 0, 4, dtype=int)
    radii = radii[:-1]
    for r, color in zip(radii, bgr_colors):
        cv2.circle(img, center, r, color, thickness=-1)

    return img

def plot_overlapping_distributions(list1, list2,
    bins=30,
    labels=('Group 1', 'Group 2'),
    colors=('tab:blue', 'tab:red'),
    alpha=0.4,
    mode='hist', show_ttest=False
):
    """
    Plot overlapping distributions of two lists.

    Parameters
    ----------
    list1, list2 : array-like
        Data values
    bins : int
        Number of bins for histograms
    labels : tuple
        Labels for the two groups
    colors : tuple
        Colors for the two groups
    alpha : float
        Transparency
    mode : str
        'hist'  -> overlapping histograms
        'kde'   -> smooth density curves
    """

    list1 = np.asarray(list1)
    list2 = np.asarray(list2)

    plt.figure(figsize=(6, 4))

    if mode == 'hist':
        plt.hist(list1, bins=bins, density=True,
                 alpha=alpha, label=labels[0], color=colors[0])
        plt.hist(list2, bins=bins, density=True,
                 alpha=alpha, label=labels[1], color=colors[1])

    elif mode == 'kde':
        xmin = min(list1.min(), list2.min())
        xmax = max(list1.max(), list2.max())
        x = np.linspace(xmin, xmax, 50)

        kde1 = gaussian_kde(list1)
        kde2 = gaussian_kde(list2)

        plt.plot(x, kde1(x), label=labels[0], color=colors[0])
        plt.plot(x, kde2(x), label=labels[1], color=colors[1])

    else:
        raise ValueError("mode must be 'hist' or 'kde'")

    # ---- Welch's t-test ----
    title = (f"{labels[0]} vs {labels[1]}")
    if show_ttest:
        t_stat, p_val = ttest_ind(list1, list2, equal_var=False)

        title = (
            f"{labels[0]} vs {labels[1]} - "
            f"Welch t-test: p = {p_val:.3e}"
        )
        

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    
    
def compare_expression_around_speckle(
    df_mean,
    treatment1,
    treatment2,
    column='Expression around speckle'
):
    vals1 = df_mean.loc[
        df_mean['Treatment'] == treatment1, column
    ].dropna().values

    vals2 = df_mean.loc[
        df_mean['Treatment'] == treatment2, column
    ].dropna().values

    if len(vals1) == 0 or len(vals2) == 0:
        raise ValueError("One of the treatments has no data.")

    plot_overlapping_distributions(
        vals1,
        vals2,
        labels=(treatment1, treatment2),
        mode='hist'
    )

def mpl_cmap_to_bgr_array(cmap):
    colors = cmap(np.linspace(0, 1, 256))[:, :3]  # RGB
    colors = (colors[:, ::-1] * 255).astype(np.uint8)  # RGB → BGR
    return colors  # shape (256, 3)

def add_colorbar(
    image,
    vmin,
    vmax,
    cmap,
    bar_height=300,
    bar_width=30,
    margin=20,
    ticks=5,
    position="top_right"
):
    """
    Adds a vertical colorbar to an image (BGR).
    Returns modified image.
    """

    img = image.copy()
    H, W = img.shape[:2]

    gradient = np.linspace(255, 0, bar_height).astype(np.uint8)
    gradient = np.tile(gradient[:, None], (1, bar_width))  # shape (H, W)
    
    cmap_array = mpl_cmap_to_bgr_array(cmap)
    
    # Map grayscale values to BGR colors
    colorbar = cmap_array[gradient]   # shape (H, W, 3)

    # ---- Decide position ----
    if position == "top_right":
        x0 = W - bar_width - margin
        y0 = margin
    elif position == "bottom_right":
        x0 = W - bar_width - margin
        y0 = H - bar_height - margin
    elif position == "top_left":
        x0 = margin
        y0 = margin
    else:  # bottom_left
        x0 = margin
        y0 = H - bar_height - margin

    # ---- Paste colorbar ----
    img[y0:y0+bar_height, x0:x0+bar_width] = colorbar

    # ---- Draw border ----
    cv2.rectangle(
        img,
        (x0, y0),
        (x0 + bar_width, y0 + bar_height),
        (255, 255, 255),
        1
    )

    # ---- Add tick labels ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    frac = 0
    label = 'Max'
    y_tick = int(y0 + frac * bar_height)
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(
        img,
        label,
        (x0 - tw - 8, y_tick + th//2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    
    frac = 1
    label = 'Min'
    y_tick = int(y0 + frac * bar_height)
    cv2.putText(
        img,
        label,
        (x0 - tw - 8, y_tick + th//2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    
    return img

def cake_plot(list_images, labels, cmap, margin=120, label_offset=50, font_scale=0.5, thickness=1):
    assert len(list_images) == len(labels)
    N = len(list_images)

    # ---------- Combine images into pie ----------
    H, W = list_images[0].shape[:2]
    C = 1 if list_images[0].ndim == 2 else list_images[0].shape[2]

    cy, cx = H / 2, W / 2
    y, x = np.indices((H, W))
    dx = x - cx
    dy = y - cy

    theta = np.arctan2(dy, dx)
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    sector_size = 2 * np.pi / N
    sector_idx = (theta // sector_size).astype(int)

    pie = np.zeros((H, W, C), dtype=list_images[0].dtype)
    for i in range(N):
        mask = sector_idx == i
        if C == 1:
            pie[..., 0][mask] = list_images[i][mask]
        else:
            pie[mask] = list_images[i][mask]

    pie = pie.squeeze()

    # ---------- Pad image ----------
    if pie.ndim == 2:
        pie = cv2.cvtColor(pie, cv2.COLOR_GRAY2BGR)

    pie_padded = cv2.copyMakeBorder(
        pie,
        margin, margin, margin, margin,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # ---------- New geometry ----------
    H2, W2 = pie_padded.shape[:2]
    cx2, cy2 = W2 // 2, H2 // 2
    radius = min(H, W) // 2

    # ---------- Draw labels ----------
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, label in enumerate(labels):
        angle = (i + 0.5) * sector_size

        r = radius + label_offset
        x_c = int(cx2 + r * np.cos(angle))
        y_c = int(cy2 + r * np.sin(angle))

        (tw, th), base = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        x0 = x_c - tw // 2
        y0 = y_c + th // 2

        # background box
        pad = 4
        cv2.rectangle(
            pie_padded,
            (x0 - pad, y0 - th - pad),
            (x0 + tw + pad, y0 + base + pad),
            (255, 255, 255),
            -1
        )

        # text
        cv2.putText(
            pie_padded,
            label,
            (x0, y0),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )
        
    pie_padded = add_colorbar(
    pie_padded,
    vmin=0,
    vmax=100,
    cmap = cmap,
    position="top_right"
)

    return pie_padded


