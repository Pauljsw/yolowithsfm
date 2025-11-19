"""
3D Visualization of Grouped Cracks in SFM Coordinate System

Visualizes:
- COLMAP sparse point cloud (gray)
- Grouped crack centroids (colored by group)
- Optional: crack extent as bounding boxes
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

from src.colmap_io import read_points3D_binary

logger = logging.getLogger(__name__)


def load_sparse_point_cloud(colmap_model_path: str, max_points: int = 50000):
    """
    Load COLMAP sparse point cloud.

    Args:
        colmap_model_path: Path to COLMAP sparse model directory
        max_points: Maximum number of points to load (for performance)

    Returns:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
    """
    points3d_path = Path(colmap_model_path) / "points3D.bin"

    logger.info(f"Loading sparse point cloud from {points3d_path}")
    points3d = read_points3D_binary(str(points3d_path))

    # Extract xyz and rgb
    all_points = []
    all_colors = []

    for pt_id, point in points3d.items():
        all_points.append(point.xyz)
        all_colors.append(point.rgb / 255.0)  # Normalize to [0, 1]

    points = np.array(all_points)
    colors = np.array(all_colors)

    # Subsample if too many points
    if len(points) > max_points:
        logger.info(f"Subsampling {len(points)} -> {max_points} points")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    logger.info(f"Loaded {len(points)} points")
    return points, colors


def load_groups(groups_path: str):
    """
    Load grouped crack data.

    Returns:
        List of group dicts
    """
    with open(groups_path) as f:
        groups = json.load(f)

    logger.info(f"Loaded {len(groups)} groups")
    return groups


def visualize_3d(
    sparse_points,
    sparse_colors,
    groups,
    output_path: str = None,
    show_extent: bool = True,
    colormap: str = 'tab20'
):
    """
    Visualize sparse point cloud + crack groups in 3D.

    Args:
        sparse_points: (N, 3) sparse point cloud
        sparse_colors: (N, 3) RGB colors
        groups: List of group dicts
        output_path: Path to save figure (optional)
        show_extent: Whether to show bounding boxes for groups
        colormap: Matplotlib colormap for groups
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sparse point cloud (gray, small, transparent)
    logger.info("Plotting sparse point cloud...")
    ax.scatter(
        sparse_points[:, 0],
        sparse_points[:, 1],
        sparse_points[:, 2],
        c='gray',
        s=0.5,
        alpha=0.3,
        label='COLMAP sparse points'
    )

    # Plot crack group centroids
    logger.info("Plotting crack groups...")
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / len(groups)) for i in range(len(groups))]

    for i, group in enumerate(groups):
        centroid = np.array(group['centroid_3d'])
        extent = np.array(group['extent_3d'])

        # Plot centroid (large, colored)
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            c=[colors[i]],
            s=200,
            marker='o',
            edgecolors='black',
            linewidths=2,
            label=f"Group {i+1} ({group['num_masks']}m, {group['num_views']}v)",
            alpha=0.8
        )

        # Plot extent as bounding box (optional)
        if show_extent:
            plot_bbox(ax, centroid, extent, color=colors[i], alpha=0.2)

    # Set labels
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Crack Groups in SFM Global Coordinate System', fontsize=16, fontweight='bold')

    # Legend (only show top 10 groups to avoid clutter)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 11:  # 1 for point cloud + 10 groups
        ax.legend(handles[:11], labels[:11], loc='upper left', fontsize=10)
    else:
        ax.legend(loc='upper left', fontsize=10)

    # Equal aspect ratio
    set_axes_equal(ax)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_bbox(ax, center, extent, color='red', alpha=0.2):
    """
    Plot a 3D bounding box.

    Args:
        ax: Matplotlib 3D axis
        center: (3,) center point
        extent: (3,) box dimensions [dx, dy, dz]
        color: Box color
        alpha: Transparency
    """
    # Define box vertices
    dx, dy, dz = extent / 2
    vertices = np.array([
        [center[0] - dx, center[1] - dy, center[2] - dz],
        [center[0] + dx, center[1] - dy, center[2] - dz],
        [center[0] + dx, center[1] + dy, center[2] - dz],
        [center[0] - dx, center[1] + dy, center[2] - dz],
        [center[0] - dx, center[1] - dy, center[2] + dz],
        [center[0] + dx, center[1] - dy, center[2] + dz],
        [center[0] + dx, center[1] + dy, center[2] + dz],
        [center[0] - dx, center[1] + dy, center[2] + dz],
    ])

    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]

    for edge in edges:
        points = vertices[edge]
        ax.plot3D(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            alpha=alpha,
            linewidth=1
        )


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale.

    Args:
        ax: Matplotlib 3D axis
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def export_to_ply(
    sparse_points,
    sparse_colors,
    groups,
    output_path: str
):
    """
    Export visualization to PLY file for viewing in CloudCompare/MeshLab.

    Args:
        sparse_points: (N, 3) sparse point cloud
        sparse_colors: (N, 3) RGB colors [0-1]
        groups: List of group dicts
        output_path: Output PLY file path
    """
    # Combine sparse points and group centroids
    all_points = [sparse_points]
    all_colors = [(sparse_colors * 255).astype(np.uint8)]

    # Add group centroids (with distinct colors)
    cmap = plt.get_cmap('tab20')
    for i, group in enumerate(groups):
        centroid = np.array(group['centroid_3d']).reshape(1, 3)
        color = (np.array(cmap(i / len(groups))[:3]) * 255).astype(np.uint8).reshape(1, 3)

        all_points.append(centroid)
        all_colors.append(color)

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    # Write PLY
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} "
                   f"{color[0]} {color[1]} {color[2]}\n")

    logger.info(f"Exported PLY to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize crack groups in SFM global coordinate system"
    )
    parser.add_argument(
        '--colmap-model', required=True,
        help='Path to COLMAP sparse model directory'
    )
    parser.add_argument(
        '--groups', required=True,
        help='Path to groups.json'
    )
    parser.add_argument(
        '--output', default='outputs/visualization_3d.png',
        help='Output image path'
    )
    parser.add_argument(
        '--export-ply',
        help='Export to PLY file for CloudCompare/MeshLab'
    )
    parser.add_argument(
        '--max-points', type=int, default=50000,
        help='Maximum number of sparse points to show'
    )
    parser.add_argument(
        '--no-extent', action='store_true',
        help='Do not show bounding boxes'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show interactive plot instead of saving'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Load data
    sparse_points, sparse_colors = load_sparse_point_cloud(
        args.colmap_model,
        max_points=args.max_points
    )
    groups = load_groups(args.groups)

    # Visualize
    output_path = None if args.show else args.output
    visualize_3d(
        sparse_points,
        sparse_colors,
        groups,
        output_path=output_path,
        show_extent=not args.no_extent
    )

    # Export PLY
    if args.export_ply:
        export_to_ply(
            sparse_points,
            sparse_colors,
            groups,
            args.export_ply
        )

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    if not args.show:
        print(f"Saved to: {args.output}")
    if args.export_ply:
        print(f"PLY exported to: {args.export_ply}")
    print("="*60)


if __name__ == '__main__':
    main()
