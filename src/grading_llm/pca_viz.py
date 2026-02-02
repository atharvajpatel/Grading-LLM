"""
PCA Visualization

Creates publication-quality 3D PCA plots of QA embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .scales import SCALE_ORDER


# Color palette for scales
SCALE_COLORS = {
    "binary": "#2196F3",       # Blue
    "ternary": "#4CAF50",      # Green
    "quaternary": "#FF9800",   # Orange
    "continuous": "#E91E63",   # Pink
}

SCALE_MARKERS = {
    "binary": "o",
    "ternary": "s",
    "quaternary": "^",
    "continuous": "D",
}


def create_pca_plot(
    coords: np.ndarray,
    scale_labels: List[str],
    explained_variance: np.ndarray,
    output_path: Path,
    title: Optional[str] = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
    show_centroids: bool = True,
    show_centroid_path: bool = True,
) -> None:
    """
    Create a 3D PCA visualization.

    Args:
        coords: Array of shape (n_samples, 3) with PCA coordinates
        scale_labels: List of scale names for each sample
        explained_variance: Array of explained variance ratios for PC1-3
        output_path: Path to save the plot
        title: Optional custom title
        figsize: Figure size
        dpi: Resolution
        show_centroids: Whether to show scale centroids
        show_centroid_path: Whether to connect centroids
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot points by scale
    for scale_name in SCALE_ORDER:
        mask = np.array(scale_labels) == scale_name
        if not any(mask):
            continue

        scale_coords = coords[mask]
        color = SCALE_COLORS.get(scale_name, "#666666")
        marker = SCALE_MARKERS.get(scale_name, "o")

        ax.scatter(
            scale_coords[:, 0],
            scale_coords[:, 1],
            scale_coords[:, 2],
            c=color,
            marker=marker,
            s=60,
            alpha=0.6,
            label=scale_name.capitalize(),
            edgecolors='white',
            linewidths=0.5
        )

    # Compute and plot centroids
    if show_centroids or show_centroid_path:
        centroids = {}
        for scale_name in SCALE_ORDER:
            mask = np.array(scale_labels) == scale_name
            if any(mask):
                centroids[scale_name] = coords[mask].mean(axis=0)

        # Plot centroids
        if show_centroids:
            for scale_name, centroid in centroids.items():
                color = SCALE_COLORS.get(scale_name, "#666666")
                ax.scatter(
                    [centroid[0]], [centroid[1]], [centroid[2]],
                    c=color,
                    marker='*',
                    s=300,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=10
                )

        # Connect centroids with path
        if show_centroid_path and len(centroids) > 1:
            path_coords = []
            for scale_name in SCALE_ORDER:
                if scale_name in centroids:
                    path_coords.append(centroids[scale_name])

            if len(path_coords) > 1:
                path_coords = np.array(path_coords)
                ax.plot(
                    path_coords[:, 0],
                    path_coords[:, 1],
                    path_coords[:, 2],
                    color='#333333',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.7,
                    zorder=5
                )

    # Labels with explained variance
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=11)
    ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=11)

    # Title
    if title is None:
        title = "Geometric Instability of QA-Embeddings Across Grading Granularity"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Adjust view angle
    ax.view_init(elev=20, azim=45)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_2d_projections(
    coords: np.ndarray,
    scale_labels: List[str],
    explained_variance: np.ndarray,
    output_dir: Path,
    prefix: str = "pca_2d"
) -> None:
    """
    Create 2D projections (PC1-PC2, PC1-PC3, PC2-PC3).

    Args:
        coords: Array of shape (n_samples, 3)
        scale_labels: List of scale names
        explained_variance: Explained variance ratios
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    projections = [
        (0, 1, "PC1", "PC2"),
        (0, 2, "PC1", "PC3"),
        (1, 2, "PC2", "PC3"),
    ]

    for pc_x, pc_y, label_x, label_y in projections:
        fig, ax = plt.subplots(figsize=(8, 7))

        for scale_name in SCALE_ORDER:
            mask = np.array(scale_labels) == scale_name
            if not any(mask):
                continue

            scale_coords = coords[mask]
            color = SCALE_COLORS.get(scale_name, "#666666")
            marker = SCALE_MARKERS.get(scale_name, "o")

            ax.scatter(
                scale_coords[:, pc_x],
                scale_coords[:, pc_y],
                c=color,
                marker=marker,
                s=80,
                alpha=0.6,
                label=scale_name.capitalize(),
                edgecolors='white',
                linewidths=0.5
            )

        ax.set_xlabel(f'{label_x} ({explained_variance[pc_x]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'{label_y} ({explained_variance[pc_y]*100:.1f}%)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{prefix}_{label_x.lower()}_{label_y.lower()}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def create_variance_plot(
    metrics: Dict,
    output_path: Path
) -> None:
    """
    Create a bar plot of variance by scale.

    Args:
        metrics: Metrics dictionary with scale_metrics
        output_path: Path to save plot
    """
    scale_metrics = metrics.get("scale_metrics", {})

    scales = []
    variances = []

    for scale_name in SCALE_ORDER:
        if scale_name in scale_metrics:
            scales.append(scale_name.capitalize())
            variances.append(scale_metrics[scale_name]["avg_variance"])

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [SCALE_COLORS.get(s.lower(), "#666666") for s in scales]
    bars = ax.bar(scales, variances, color=colors, edgecolor='white', linewidth=1.5)

    ax.set_ylabel('Average Variance', fontsize=11)
    ax.set_xlabel('Grading Scale', fontsize=11)
    ax.set_title('Response Variance by Grading Granularity', fontsize=12, fontweight='bold')

    # Add value labels on bars
    for bar, var in zip(bars, variances):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{var:.4f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
