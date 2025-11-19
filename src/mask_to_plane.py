"""
Mask-to-Plane Transformation (Phase 3 Alternative)

Instead of sparse point cloud overlay with DBSCAN, this module:
1. Backprojects each YOLO mask to 3D using depth + SFM camera pose
2. Fits a plane to each mask's 3D point cloud (planar crack assumption)
3. Outputs plane parameters for mask-based clustering

Assumptions:
- Cracks lie on planar surfaces (walls, floors, ceilings)
- Depth images are available and aligned to RGB
- SFM provides camera poses in global coordinates

Output format per mask:
{
    "image": "image_name.jpg",
    "mask_idx": 0,
    "confidence": 0.85,
    "plane": {
        "normal": [a, b, c],  # Unit normal vector
        "d": d,                # Distance from origin (ax + by + cz + d = 0)
        "center": [x, y, z],   # Centroid of 3D points
        "area_3d": 123.45,     # 3D surface area (m²)
        "num_points": 5000     # Number of valid 3D points
    },
    "quality": {
        "planarity": 0.95,     # How well points fit the plane (0-1)
        "depth_coverage": 0.80 # % of mask pixels with valid depth
    }
}
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from src.colmap_io import read_images_binary, Image as ColmapImage, qvec2rotmat

logger = logging.getLogger(__name__)


@dataclass
class PlaneFit:
    """Result of plane fitting for a mask"""
    normal: np.ndarray      # (3,) unit normal vector
    d: float                # Distance from origin
    center: np.ndarray      # (3,) centroid
    area_3d: float          # 3D surface area in m²
    num_points: int         # Number of valid 3D points
    planarity: float        # Fit quality metric (0-1)
    depth_coverage: float   # % of mask pixels with valid depth


class MaskToPlaneConverter:
    """Convert 2D YOLO masks to 3D plane representations"""

    def __init__(
        self,
        rgb_calib_path: str,
        depth_calib_path: str,
        extrinsic_path: str,
        colmap_model_path: str,
        depth_scale: float = 1000.0  # mm to meters
    ):
        """
        Args:
            rgb_calib_path: Path to RGB camera calibration JSON
            depth_calib_path: Path to depth camera calibration JSON
            extrinsic_path: Path to depth-to-RGB extrinsic JSON
            colmap_model_path: Path to COLMAP sparse model directory
            depth_scale: Scale factor to convert depth values to meters
        """
        # Load calibration
        with open(rgb_calib_path) as f:
            rgb_calib = json.load(f)
        with open(depth_calib_path) as f:
            depth_calib = json.load(f)
        with open(extrinsic_path) as f:
            extrinsic = json.load(f)

        self.K_rgb = np.array(rgb_calib['K'])
        self.D_rgb = np.array(rgb_calib['D'])
        self.rgb_size = (rgb_calib['width'], rgb_calib['height'])

        self.K_depth = np.array(depth_calib['K'])
        self.depth_size = (depth_calib['width'], depth_calib['height'])

        self.depth_scale = depth_scale

        # Load COLMAP camera poses
        self.colmap_images = read_images_binary(
            Path(colmap_model_path) / "images.bin"
        )
        logger.info(f"Loaded {len(self.colmap_images)} camera poses from COLMAP")

    def backproject_mask_to_3d(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        camera_pose: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backproject mask pixels to 3D global coordinates.

        Args:
            mask: (H, W) boolean mask from YOLO
            depth: (H_d, W_d) depth image (aligned to RGB)
            camera_pose: {"R": (3,3), "t": (3,)} from COLMAP

        Returns:
            points_3d: (N, 3) 3D points in global coordinates
            colors: (N, 3) RGB colors for visualization (optional)
        """
        H, W = mask.shape
        H_d, W_d = self.depth_size

        # Get mask pixel coordinates in RGB frame
        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # Scale coordinates to depth image resolution
        x_depth = (x_coords * W_d / W).astype(int)
        y_depth = (y_coords * H_d / H).astype(int)

        # Clip to valid depth image bounds
        valid_bounds = (
            (x_depth >= 0) & (x_depth < W_d) &
            (y_depth >= 0) & (y_depth < H_d)
        )
        x_depth = x_depth[valid_bounds]
        y_depth = y_depth[valid_bounds]
        x_rgb = x_coords[valid_bounds]
        y_rgb = y_coords[valid_bounds]

        # Get depth values
        depth_values = depth[y_depth, x_depth].astype(np.float32) / self.depth_scale

        # Filter valid depth (e.g., 0.1m to 10m)
        valid_depth = (depth_values > 0.1) & (depth_values < 10.0)
        x_rgb = x_rgb[valid_depth]
        y_rgb = y_rgb[valid_depth]
        depth_values = depth_values[valid_depth]

        if len(depth_values) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # Backproject to 3D in camera frame
        fx, fy = self.K_rgb[0, 0], self.K_rgb[1, 1]
        cx, cy = self.K_rgb[0, 2], self.K_rgb[1, 2]

        x_cam = (x_rgb - cx) * depth_values / fx
        y_cam = (y_rgb - cy) * depth_values / fy
        z_cam = depth_values

        points_camera = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

        # Transform to global coordinates
        # COLMAP uses: P_world = R^T * (P_cam - t)
        R = camera_pose["R"]
        t = camera_pose["t"]

        points_global = (R.T @ (points_camera - t).T).T  # (N, 3)

        # Placeholder for colors (can extend later)
        colors = np.zeros_like(points_global)

        return points_global, colors

    def fit_plane_pca(self, points: np.ndarray) -> Optional[PlaneFit]:
        """
        Fit a plane to 3D points using PCA.

        The plane normal is the eigenvector with smallest eigenvalue.

        Args:
            points: (N, 3) 3D points

        Returns:
            PlaneFit object or None if fitting fails
        """
        if len(points) < 10:  # Need minimum points
            return None

        # Center points
        center = points.mean(axis=0)
        points_centered = points - center

        # PCA: eigenvector with smallest eigenvalue is normal
        cov = points_centered.T @ points_centered / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal is the eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]  # (3,)

        # Ensure consistent orientation (normal points "outward")
        if normal[2] < 0:  # Assume Z-up convention
            normal = -normal

        # Plane equation: normal · (P - center) = 0
        # Expand: normal · P - normal · center = 0
        # Standard form: ax + by + cz + d = 0
        d = -np.dot(normal, center)

        # Compute planarity (inverse of normalized variance along normal)
        distances = points_centered @ normal
        planarity = 1.0 - np.std(distances) / (np.linalg.norm(points_centered, axis=1).std() + 1e-6)
        planarity = np.clip(planarity, 0.0, 1.0)

        # Estimate 3D surface area (project to plane and compute convex hull area)
        # This is approximate - for simplicity, use bounding box area
        u = eigenvectors[:, 1]  # Second principal direction
        v = eigenvectors[:, 2]  # Third principal direction

        coords_u = points_centered @ u
        coords_v = points_centered @ v

        area_3d = (coords_u.max() - coords_u.min()) * (coords_v.max() - coords_v.min())

        return PlaneFit(
            normal=normal,
            d=d,
            center=center,
            area_3d=area_3d,
            num_points=len(points),
            planarity=planarity,
            depth_coverage=1.0  # Will be set externally
        )

    def process_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        image_name: str,
        mask_idx: int,
        confidence: float
    ) -> Optional[Dict]:
        """
        Process a single mask: backproject to 3D and fit plane.

        Args:
            mask: (H, W) boolean mask
            depth: (H_d, W_d) depth image
            image_name: Name of source image (for COLMAP lookup)
            mask_idx: Index of mask in image
            confidence: YOLO confidence score

        Returns:
            Dictionary with mask plane data or None if processing fails
        """
        # Find COLMAP image by name
        colmap_image = None
        for img_id, img in self.colmap_images.items():
            if img.name == image_name:
                colmap_image = img
                break

        if colmap_image is None:
            logger.warning(f"Image {image_name} not found in COLMAP model")
            return None

        # Get camera pose
        # COLMAP stores: R_world_to_cam (qvec), t_world_to_cam (tvec)
        R_world_to_cam = qvec2rotmat(colmap_image.qvec)
        t_world_to_cam = colmap_image.tvec

        # Convert to camera-to-world (needed for backprojection)
        R_cam_to_world = R_world_to_cam.T
        t_cam_to_world = -R_world_to_cam.T @ t_world_to_cam

        camera_pose = {
            "R": R_cam_to_world,
            "t": t_cam_to_world
        }

        # Backproject mask to 3D
        points_3d, _ = self.backproject_mask_to_3d(mask, depth, camera_pose)

        if len(points_3d) == 0:
            logger.warning(
                f"[{image_name}] Mask {mask_idx}: No valid 3D points (no depth coverage)"
            )
            return None

        # Compute depth coverage
        total_mask_pixels = mask.sum()
        depth_coverage = len(points_3d) / total_mask_pixels if total_mask_pixels > 0 else 0.0

        # Fit plane
        plane_fit = self.fit_plane_pca(points_3d)

        if plane_fit is None:
            logger.warning(
                f"[{image_name}] Mask {mask_idx}: Plane fitting failed "
                f"(only {len(points_3d)} points)"
            )
            return None

        plane_fit.depth_coverage = depth_coverage

        # Return structured result
        return {
            "image": image_name,
            "mask_idx": mask_idx,
            "confidence": float(confidence),
            "plane": {
                "normal": plane_fit.normal.tolist(),
                "d": float(plane_fit.d),
                "center": plane_fit.center.tolist(),
                "area_3d": float(plane_fit.area_3d),
                "num_points": int(plane_fit.num_points)
            },
            "quality": {
                "planarity": float(plane_fit.planarity),
                "depth_coverage": float(plane_fit.depth_coverage)
            }
        }

    def process_image(
        self,
        image_path: str,
        depth_path: str,
        yolo_results_path: str
    ) -> List[Dict]:
        """
        Process all masks in an image.

        Args:
            image_path: Path to RGB image
            depth_path: Path to aligned depth image
            yolo_results_path: Path to YOLO detection JSON

        Returns:
            List of mask plane dictionaries
        """
        image_name = Path(image_path).name

        # Load YOLO results
        with open(yolo_results_path) as f:
            yolo_results = json.load(f)

        # Load depth
        depth = cv2.imread(depth_path, -1)
        if depth is None:
            logger.error(f"Cannot load depth image: {depth_path}")
            return []

        # Load image size for mask reconstruction
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot load RGB image: {image_path}")
            return []
        H, W = img.shape[:2]

        # Process each detection
        results = []
        for det_idx, detection in enumerate(yolo_results):
            if 'segmentation' not in detection:
                continue

            # Reconstruct mask from polygon
            polygon = np.array(detection['segmentation'], dtype=np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            mask = mask.astype(bool)

            confidence = detection.get('confidence', 1.0)

            # Process mask
            result = self.process_mask(
                mask=mask,
                depth=depth,
                image_name=image_name,
                mask_idx=det_idx,
                confidence=confidence
            )

            if result is not None:
                results.append(result)
                logger.info(
                    f"[{image_name}] Mask {det_idx}: "
                    f"planarity={result['quality']['planarity']:.3f}, "
                    f"coverage={result['quality']['depth_coverage']:.1%}, "
                    f"points={result['plane']['num_points']}"
                )

        return results


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert YOLO masks to 3D planes using depth + SFM"
    )
    parser.add_argument(
        '--rgb-calib', required=True,
        help='Path to RGB camera calibration JSON'
    )
    parser.add_argument(
        '--depth-calib', required=True,
        help='Path to depth camera calibration JSON'
    )
    parser.add_argument(
        '--extrinsic', required=True,
        help='Path to depth-to-RGB extrinsic JSON'
    )
    parser.add_argument(
        '--colmap-model', required=True,
        help='Path to COLMAP sparse model directory'
    )
    parser.add_argument(
        '--image', required=True,
        help='Path to RGB image'
    )
    parser.add_argument(
        '--depth', required=True,
        help='Path to aligned depth image'
    )
    parser.add_argument(
        '--yolo-results', required=True,
        help='Path to YOLO detection JSON'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output JSON path for plane parameters'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Create converter
    converter = MaskToPlaneConverter(
        rgb_calib_path=args.rgb_calib,
        depth_calib_path=args.depth_calib,
        extrinsic_path=args.extrinsic,
        colmap_model_path=args.colmap_model
    )

    # Process image
    results = converter.process_image(
        image_path=args.image,
        depth_path=args.depth,
        yolo_results_path=args.yolo_results
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} plane fits to {output_path}")


if __name__ == '__main__':
    main()
