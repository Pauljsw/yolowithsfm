"""
Simple 3D Visualization: SFM Point Cloud + Grouped Crack Masks

Uses Open3D for reliable visualization.
Shows:
1. COLMAP sparse point cloud (gray)
2. Each group's masks as colored 3D points
"""

import numpy as np
import json
import cv2
from pathlib import Path
import logging
from typing import List, Dict
import open3d as o3d

from src.colmap_io import read_images_binary, read_points3D_binary, qvec2rotmat

logger = logging.getLogger(__name__)


def load_sfm_point_cloud(colmap_model_path: str, max_points: int = 50000):
    """Load COLMAP sparse point cloud"""
    points3d_path = Path(colmap_model_path) / "points3D.bin"

    logger.info(f"Loading sparse point cloud...")
    points3d = read_points3D_binary(str(points3d_path))

    points = []
    colors = []

    for pt_id, point in points3d.items():
        points.append(point.xyz)
        colors.append(point.rgb / 255.0)

    points = np.array(points)
    colors = np.array(colors)

    # Subsample
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    logger.info(f"Loaded {len(points)} sparse points")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors * 0.5)  # Darker for background

    return pcd


def backproject_mask_points(
    mask: np.ndarray,
    depth: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    depth_size: tuple,
    R: np.ndarray,
    t: np.ndarray,
    depth_scale: float = 1000.0,
    sample_rate: int = 5
):
    """Backproject mask to 3D points"""
    H, W = mask.shape
    H_d, W_d = depth_size

    # Get mask pixels
    y_coords, x_coords = np.where(mask)

    if len(x_coords) == 0:
        return np.zeros((0, 3))

    # Subsample
    if sample_rate > 1:
        indices = np.arange(0, len(x_coords), sample_rate)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]

    # Scale to depth
    x_depth = (x_coords * W_d / W).astype(int)
    y_depth = (y_coords * H_d / H).astype(int)

    # Bounds check
    valid = (
        (x_depth >= 0) & (x_depth < W_d) &
        (y_depth >= 0) & (y_depth < H_d)
    )
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    x_depth = x_depth[valid]
    y_depth = y_depth[valid]

    # Get depth
    depth_values = depth[y_depth, x_depth].astype(np.float32) / depth_scale
    valid_depth = (depth_values > 0.1) & (depth_values < 10.0)

    x_coords = x_coords[valid_depth]
    y_coords = y_coords[valid_depth]
    depth_values = depth_values[valid_depth]

    if len(depth_values) == 0:
        return np.zeros((0, 3))

    # Backproject
    fx, fy = K_rgb[0, 0], K_rgb[1, 1]
    cx, cy = K_rgb[0, 2], K_rgb[1, 2]

    x_cam = (x_coords - cx) * depth_values / fx
    y_cam = (y_coords - cy) * depth_values / fy
    z_cam = depth_values

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # To global
    points_global = (R.T @ (points_cam - t).T).T

    return points_global


def visualize_groups_3d(
    groups_path: str,
    masks_3d_dir: str,
    rgb_dir: str,
    depth_dir: str,
    rgb_calib_path: str,
    depth_calib_path: str,
    colmap_model_path: str,
    max_sparse_points: int = 50000,
    sample_rate: int = 10,
    min_views: int = 1,
    save_path: str = None
):
    """
    Visualize groups with Open3D

    Args:
        groups_path: groups.json
        masks_3d_dir: Directory with mask 3D JSONs
        rgb_dir: RGB images
        depth_dir: Depth images
        rgb_calib_path: RGB calibration
        depth_calib_path: Depth calibration
        colmap_model_path: COLMAP model
        max_sparse_points: Max sparse points to show
        sample_rate: Point sampling rate (higher = faster, less dense)
        min_views: Minimum views to include group
        save_path: Optional path to save PLY
    """
    # Load calibration
    with open(rgb_calib_path) as f:
        rgb_calib = json.load(f)
    with open(depth_calib_path) as f:
        depth_calib = json.load(f)

    K_rgb = np.array(rgb_calib['K'])
    K_depth = np.array(depth_calib['K'])
    depth_size = (depth_calib['width'], depth_calib['height'])
    depth_scale = 1000.0

    # Load COLMAP
    colmap_images = read_images_binary(Path(colmap_model_path) / "images.bin")

    # Load groups
    with open(groups_path) as f:
        groups = json.load(f)

    logger.info(f"Loaded {len(groups)} groups")

    # Filter by views
    groups = [g for g in groups if g['num_views'] >= min_views]
    logger.info(f"After filtering (views>={min_views}): {len(groups)} groups")

    # Load sparse point cloud
    sparse_pcd = load_sfm_point_cloud(colmap_model_path, max_sparse_points)

    # Color palette
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20')

    geometries = [sparse_pcd]

    # Process each group
    for group_idx, group in enumerate(groups):
        logger.info(f"Processing group {group_idx + 1}/{len(groups)}...")

        color = np.array(cmap(group_idx / max(len(groups), 1))[:3])

        all_points = []

        # Process each mask in group
        for mask_info in group['masks']:
            image_name = mask_info['image']
            mask_idx = mask_info['mask_idx']

            # Paths
            rgb_path = Path(rgb_dir) / image_name
            timestamp = image_name.replace("camera_RGB_", "").replace(".png", "")
            depth_path = Path(depth_dir) / f"camera_DPT_{timestamp}.png"
            yolo_path = Path(depth_dir).parent / "yolo_masks" / f"camera_RGB_{timestamp}.json"

            if not all([rgb_path.exists(), depth_path.exists(), yolo_path.exists()]):
                continue

            # Load YOLO mask
            with open(yolo_path) as f:
                yolo_data = json.load(f)

            detections = yolo_data['masks'] if 'masks' in yolo_data else yolo_data

            if mask_idx >= len(detections):
                continue

            detection = detections[mask_idx]

            # Reconstruct mask
            img = cv2.imread(str(rgb_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            if 'polygon' in detection:
                polygon = np.array(detection['polygon'], dtype=np.int32)
            else:
                continue

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            mask = mask.astype(bool)

            # Load depth
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue

            # Get camera pose
            colmap_image = None
            for img_id, img_data in colmap_images.items():
                if img_data.name == image_name:
                    colmap_image = img_data
                    break

            if colmap_image is None:
                continue

            R_w2c = qvec2rotmat(colmap_image.qvec)
            t_w2c = colmap_image.tvec
            R = R_w2c.T
            t = -R_w2c.T @ t_w2c

            # Backproject
            points_3d = backproject_mask_points(
                mask, depth, K_rgb, K_depth, depth_size, R, t,
                depth_scale, sample_rate
            )

            if len(points_3d) > 0:
                all_points.append(points_3d)

        # Create point cloud for this group
        if len(all_points) > 0:
            all_points = np.vstack(all_points)

            group_pcd = o3d.geometry.PointCloud()
            group_pcd.points = o3d.utility.Vector3dVector(all_points)
            group_pcd.paint_uniform_color(color)

            geometries.append(group_pcd)

            logger.info(f"  Group {group_idx + 1}: {len(all_points)} points")

    logger.info(f"Total geometries: {len(geometries)}")

    # Save if requested
    if save_path:
        # Merge all into one point cloud
        merged = o3d.geometry.PointCloud()
        for geom in geometries:
            merged += geom

        o3d.io.write_point_cloud(save_path, merged)
        logger.info(f"Saved to {save_path}")

    # Visualize
    logger.info("Opening visualization window...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="SFM Point Cloud + Crack Groups",
        width=1600,
        height=1200,
        left=50,
        top=50
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize crack groups with SFM point cloud (Open3D)"
    )
    parser.add_argument('--groups', required=True)
    parser.add_argument('--masks-3d-dir', required=True)
    parser.add_argument('--rgb-dir', required=True)
    parser.add_argument('--depth-dir', required=True)
    parser.add_argument('--rgb-calib', required=True)
    parser.add_argument('--depth-calib', required=True)
    parser.add_argument('--colmap-model', required=True)
    parser.add_argument('--max-sparse-points', type=int, default=50000)
    parser.add_argument('--sample-rate', type=int, default=10,
                       help='Mask point sampling (10=every 10th pixel)')
    parser.add_argument('--min-views', type=int, default=2,
                       help='Minimum views to show group')
    parser.add_argument('--save', help='Save to PLY file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    visualize_groups_3d(
        groups_path=args.groups,
        masks_3d_dir=args.masks_3d_dir,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        rgb_calib_path=args.rgb_calib,
        depth_calib_path=args.depth_calib,
        colmap_model_path=args.colmap_model,
        max_sparse_points=args.max_sparse_points,
        sample_rate=args.sample_rate,
        min_views=args.min_views,
        save_path=args.save
    )


if __name__ == '__main__':
    main()
