"""
Lightweight Mask-to-3D Transformation

Converts YOLO masks to 3D using depth (lightweight approach):
- Uses mask centroid + median depth (not full point cloud)
- Fast and memory efficient
- No plane fitting required

Output per mask:
{
    "image": "camera_RGB_*.png",
    "mask_idx": 0,
    "confidence": 0.85,
    "centroid_2d": [x, y],          # 2D centroid in image space
    "centroid_3d": [x, y, z],       # 3D centroid in global coordinates
    "depth_median": 2.45,            # Median depth value (m)
    "depth_std": 0.12,               # Depth variation (m)
    "mask_area_2d": 15000,           # Mask area in pixels
    "bbox_2d": [x1, y1, x2, y2],     # 2D bounding box
    "bbox_3d_size": [w, h, d]        # Approximate 3D size (m)
}
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from src.colmap_io import read_images_binary, qvec2rotmat

logger = logging.getLogger(__name__)


class MaskTo3DConverter:
    """Convert 2D YOLO masks to lightweight 3D representation"""

    def __init__(
        self,
        rgb_calib_path: str,
        depth_calib_path: str,
        colmap_model_path: str,
        depth_scale: float = 1000.0  # mm to meters
    ):
        """
        Args:
            rgb_calib_path: Path to RGB camera calibration JSON
            depth_calib_path: Path to depth camera calibration JSON
            colmap_model_path: Path to COLMAP sparse model directory
            depth_scale: Scale factor to convert depth values to meters
        """
        # Load calibration
        with open(rgb_calib_path) as f:
            rgb_calib = json.load(f)
        with open(depth_calib_path) as f:
            depth_calib = json.load(f)

        self.K_rgb = np.array(rgb_calib['K'])
        self.rgb_size = (rgb_calib['width'], rgb_calib['height'])

        self.K_depth = np.array(depth_calib['K'])
        self.depth_size = (depth_calib['width'], depth_calib['height'])

        self.depth_scale = depth_scale

        # Load COLMAP camera poses
        self.colmap_images = read_images_binary(
            Path(colmap_model_path) / "images.bin"
        )
        logger.info(f"Loaded {len(self.colmap_images)} camera poses from COLMAP")

    def project_to_3d(
        self,
        x: float,
        y: float,
        depth: float,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Project 2D point to 3D global coordinates.

        Args:
            x, y: 2D coordinates in image
            depth: Depth value in meters
            K: Camera intrinsic matrix
            R: Camera rotation (cam to world)
            t: Camera translation (cam to world)

        Returns:
            3D point in global coordinates
        """
        # Backproject to camera frame
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (x - cx) * depth / fx
        y_cam = (y - cy) * depth / fy
        z_cam = depth

        point_cam = np.array([x_cam, y_cam, z_cam])

        # Transform to global coordinates
        # P_world = R^T * (P_cam - t)
        point_global = R.T @ (point_cam - t)

        return point_global

    def process_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        image_name: str,
        mask_idx: int,
        confidence: float
    ) -> Optional[Dict]:
        """
        Process a single mask: compute 3D centroid and metadata.

        Args:
            mask: (H, W) boolean mask
            depth: (H_d, W_d) depth image
            image_name: Name of source image (for COLMAP lookup)
            mask_idx: Index of mask in image
            confidence: YOLO confidence score

        Returns:
            Dictionary with mask 3D data or None if processing fails
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
        R_world_to_cam = qvec2rotmat(colmap_image.qvec)
        t_world_to_cam = colmap_image.tvec

        # Convert to camera-to-world
        R_cam_to_world = R_world_to_cam.T
        t_cam_to_world = -R_world_to_cam.T @ t_world_to_cam

        # Get mask properties
        H, W = mask.shape
        H_d, W_d = self.depth_size

        # Compute 2D centroid
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            logger.warning(f"[{image_name}] Mask {mask_idx}: Empty mask")
            return None

        centroid_2d_x = float(x_coords.mean())
        centroid_2d_y = float(y_coords.mean())

        # Scale to depth image coordinates
        x_depth = (x_coords * W_d / W).astype(int)
        y_depth = (y_coords * H_d / H).astype(int)

        # Clip to valid depth image bounds
        valid_bounds = (
            (x_depth >= 0) & (x_depth < W_d) &
            (y_depth >= 0) & (y_depth < H_d)
        )
        x_depth = x_depth[valid_bounds]
        y_depth = y_depth[valid_bounds]

        if len(x_depth) == 0:
            logger.warning(f"[{image_name}] Mask {mask_idx}: No valid depth coordinates")
            return None

        # Get depth values
        depth_values = depth[y_depth, x_depth].astype(np.float32) / self.depth_scale

        # Filter valid depth (e.g., 0.1m to 10m)
        valid_depth = (depth_values > 0.1) & (depth_values < 10.0)
        depth_values = depth_values[valid_depth]

        if len(depth_values) == 0:
            logger.warning(f"[{image_name}] Mask {mask_idx}: No valid depth values")
            return None

        # Compute depth statistics
        depth_median = float(np.median(depth_values))
        depth_std = float(np.std(depth_values))

        # Project centroid to 3D using median depth
        centroid_3d = self.project_to_3d(
            centroid_2d_x,
            centroid_2d_y,
            depth_median,
            self.K_rgb,
            R_cam_to_world,
            t_cam_to_world
        )

        # Compute 2D bounding box
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Estimate 3D bounding box size (rough approximation)
        # Project bbox corners and compute size
        corners_3d = []
        for corner_x, corner_y in [(x_min, y_min), (x_max, y_max)]:
            corner_3d = self.project_to_3d(
                corner_x, corner_y, depth_median,
                self.K_rgb, R_cam_to_world, t_cam_to_world
            )
            corners_3d.append(corner_3d)

        bbox_3d_diagonal = np.linalg.norm(corners_3d[1] - corners_3d[0])
        bbox_3d_size = [
            float(bbox_3d_diagonal * 0.7),  # Approximate width
            float(bbox_3d_diagonal * 0.7),  # Approximate height
            float(depth_std * 2.0)          # Approximate depth extent
        ]

        # Return structured result
        return {
            "image": image_name,
            "mask_idx": mask_idx,
            "confidence": float(confidence),
            "centroid_2d": [centroid_2d_x, centroid_2d_y],
            "centroid_3d": centroid_3d.tolist(),
            "depth_median": depth_median,
            "depth_std": depth_std,
            "mask_area_2d": int(mask.sum()),
            "bbox_2d": [x_min, y_min, x_max, y_max],
            "bbox_3d_size": bbox_3d_size,
            "depth_coverage": float(len(depth_values) / len(x_coords))
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
            List of mask 3D dictionaries
        """
        image_name = Path(image_path).name

        # Load YOLO results
        with open(yolo_results_path) as f:
            yolo_data = json.load(f)

        # Handle different YOLO output formats
        if 'masks' in yolo_data:
            # Format: {"masks": [...]}
            detections = yolo_data['masks']
        elif isinstance(yolo_data, list):
            # Format: [...]
            detections = yolo_data
        else:
            logger.error(f"Unknown YOLO format in {yolo_results_path}")
            return []

        # Load depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
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
        for det_idx, detection in enumerate(detections):
            # Handle different polygon formats
            if 'polygon' in detection:
                # Format: "polygon": [[x1, y1], [x2, y2], ...]
                polygon_coords = detection['polygon']
                if len(polygon_coords) > 0 and isinstance(polygon_coords[0], list):
                    # Convert [[x, y], ...] to flat array for cv2.fillPoly
                    polygon = np.array(polygon_coords, dtype=np.int32)
                else:
                    # Already flat [x1, y1, x2, y2, ...]
                    polygon = np.array(polygon_coords, dtype=np.int32).reshape(-1, 2)
            elif 'segmentation' in detection:
                # Format: "segmentation": [x1, y1, x2, y2, ...] or [[x1, y1], ...]
                seg = detection['segmentation']
                if isinstance(seg[0], list):
                    polygon = np.array(seg, dtype=np.int32)
                else:
                    polygon = np.array(seg, dtype=np.int32).reshape(-1, 2)
            else:
                logger.warning(f"[{image_name}] Detection {det_idx}: No polygon/segmentation data")
                continue

            # Reconstruct mask from polygon
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            mask = mask.astype(bool)

            confidence = detection.get('confidence', detection.get('score', 1.0))

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
                    f"centroid_3d={result['centroid_3d']}, "
                    f"depth={result['depth_median']:.2f}m"
                )

        return results


def process_batch(
    rgb_dir: str,
    depth_dir: str,
    yolo_dir: str,
    output_dir: str,
    rgb_calib: str,
    depth_calib: str,
    colmap_model: str
):
    """
    Process all images in batch.

    Args:
        rgb_dir: Directory with RGB images (camera_RGB_*.png)
        depth_dir: Directory with depth images (camera_DPT_*.png)
        yolo_dir: Directory with YOLO results (camera_RGB_*.json)
        output_dir: Output directory for 3D masks
        rgb_calib: RGB camera calibration path
        depth_calib: Depth camera calibration path
        colmap_model: COLMAP model directory
    """
    from glob import glob
    from tqdm import tqdm

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create converter (loads COLMAP once)
    logger.info("Loading COLMAP model and calibration...")
    converter = MaskTo3DConverter(
        rgb_calib_path=rgb_calib,
        depth_calib_path=depth_calib,
        colmap_model_path=colmap_model
    )

    # Find all RGB images
    rgb_files = sorted(glob(f"{rgb_dir}/camera_RGB_*.png"))
    logger.info(f"Found {len(rgb_files)} RGB images")

    # Process statistics
    processed = 0
    skipped = 0
    failed = 0

    # Process each image
    for rgb_path in tqdm(rgb_files, desc="Processing images"):
        filename = Path(rgb_path).name
        timestamp = filename.replace("camera_RGB_", "").replace(".png", "")

        depth_path = f"{depth_dir}/camera_DPT_{timestamp}.png"
        yolo_path = f"{yolo_dir}/camera_RGB_{timestamp}.json"
        output_path = f"{output_dir}/camera_RGB_{timestamp}.json"

        # Check if files exist
        if not Path(depth_path).exists():
            logger.warning(f"Missing depth: {depth_path}")
            skipped += 1
            continue

        if not Path(yolo_path).exists():
            logger.warning(f"Missing YOLO: {yolo_path}")
            skipped += 1
            continue

        # Process image
        try:
            results = converter.process_image(
                image_path=rgb_path,
                depth_path=depth_path,
                yolo_results_path=yolo_path
            )

            # Save results
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            processed += 1

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            failed += 1

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files:  {len(rgb_files)}")
    print(f"✅ Processed:  {processed}")
    print(f"⚠️  Skipped:    {skipped}")
    print(f"❌ Failed:     {failed}")
    print(f"\nOutput: {output_dir}/")
    print("="*60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert YOLO masks to lightweight 3D representation"
    )
    parser.add_argument('--rgb-calib', required=True, help='RGB camera calibration JSON')
    parser.add_argument('--depth-calib', required=True, help='Depth camera calibration JSON')
    parser.add_argument('--colmap-model', required=True, help='COLMAP sparse model directory')

    # Single image mode
    parser.add_argument('--image', help='Single RGB image to process')
    parser.add_argument('--depth', help='Single depth image')
    parser.add_argument('--yolo-results', help='Single YOLO result JSON')
    parser.add_argument('--output', help='Output JSON path')

    # Batch mode
    parser.add_argument('--rgb-dir', help='Directory with RGB images')
    parser.add_argument('--depth-dir', help='Directory with depth images')
    parser.add_argument('--yolo-dir', help='Directory with YOLO results')
    parser.add_argument('--output-dir', help='Output directory')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Batch mode
    if args.rgb_dir:
        if not all([args.depth_dir, args.yolo_dir, args.output_dir]):
            parser.error("Batch mode requires: --rgb-dir, --depth-dir, --yolo-dir, --output-dir")

        process_batch(
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            yolo_dir=args.yolo_dir,
            output_dir=args.output_dir,
            rgb_calib=args.rgb_calib,
            depth_calib=args.depth_calib,
            colmap_model=args.colmap_model
        )

    # Single image mode
    elif args.image:
        if not all([args.depth, args.yolo_results, args.output]):
            parser.error("Single mode requires: --image, --depth, --yolo-results, --output")

        # Create converter
        converter = MaskTo3DConverter(
            rgb_calib_path=args.rgb_calib,
            depth_calib_path=args.depth_calib,
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

        logger.info(f"Saved {len(results)} masks to {output_path}")

    else:
        parser.error("Must specify either batch mode (--rgb-dir) or single mode (--image)")


if __name__ == '__main__':
    main()
