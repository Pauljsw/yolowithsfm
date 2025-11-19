"""
Depth to Color (D2C) Alignment for Orbbec Femto Bolt

Aligns depth images (512x512) to color images (3840x2160) using calibration parameters.
Uses geometric transformation with intrinsic and extrinsic calibration.

Usage:
    # Single image
    python -m src.align_depth_to_rgb \
        --depth-image data/depth/sample.png \
        --output aligned_depth.png

    # Batch processing
    python -m src.align_depth_to_rgb \
        --depth-dir data/depth \
        --output-dir data/depth_upsampled \
        --visualize
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DepthToColorAligner:
    """Align depth images to color camera frame using intrinsic and extrinsic parameters"""

    def __init__(self,
                 rgb_calib_path: str = 'calib/rgb_camera_info.json',
                 depth_calib_path: str = 'calib/depth_camera_info.json',
                 extrinsic_path: str = 'calib/extrinsic_depth_to_color.json'):
        """
        Initialize the aligner with calibration parameters

        Args:
            rgb_calib_path: Path to RGB camera intrinsic parameters JSON
            depth_calib_path: Path to Depth camera intrinsic parameters JSON
            extrinsic_path: Path to Depth-to-Color extrinsic transformation JSON
        """
        # Load calibration parameters
        with open(rgb_calib_path, 'r') as f:
            self.rgb_calib = json.load(f)

        with open(depth_calib_path, 'r') as f:
            self.depth_calib = json.load(f)

        with open(extrinsic_path, 'r') as f:
            self.extrinsic = json.load(f)

        # Parse RGB camera parameters
        self.rgb_K = np.array(self.rgb_calib['K'], dtype=np.float32)
        self.rgb_D = np.array(self.rgb_calib['D'], dtype=np.float32)
        self.rgb_width = self.rgb_calib['width']
        self.rgb_height = self.rgb_calib['height']

        # Parse Depth camera parameters
        self.depth_K = np.array(self.depth_calib['K'], dtype=np.float32)
        self.depth_D = np.array(self.depth_calib['D'], dtype=np.float32)
        self.depth_width = self.depth_calib['width']
        self.depth_height = self.depth_calib['height']

        # Parse extrinsic parameters (Depth to Color transformation)
        self.R = np.array(self.extrinsic['R'], dtype=np.float32)
        self.t = np.array(self.extrinsic['t'], dtype=np.float32).reshape(3, 1)

        logger.info("Initialized D2C Aligner:")
        logger.info(f"  RGB: {self.rgb_width}×{self.rgb_height}")
        logger.info(f"  Depth: {self.depth_width}×{self.depth_height}")
        logger.info(f"  Baseline: {self.extrinsic.get('baseline_mm', 'N/A')} mm")

    def align_depth_to_color(self,
                             depth_image: np.ndarray,
                             depth_scale: float = 1.0) -> np.ndarray:
        """
        Align depth image to color camera frame using geometric transformation

        Args:
            depth_image: Depth image (512×512), values in millimeters
            depth_scale: Scale factor to convert depth values to meters (default: 1.0 for mm)

        Returns:
            Aligned depth image in color camera resolution (3840×2160), uint16 mm
        """
        # Create output aligned depth image
        aligned_depth = np.zeros((self.rgb_height, self.rgb_width), dtype=np.uint16)

        # Get depth camera intrinsics
        fx_d = self.depth_K[0, 0]
        fy_d = self.depth_K[1, 1]
        cx_d = self.depth_K[0, 2]
        cy_d = self.depth_K[1, 2]

        # Get color camera intrinsics
        fx_c = self.rgb_K[0, 0]
        fy_c = self.rgb_K[1, 1]
        cx_c = self.rgb_K[0, 2]
        cy_c = self.rgb_K[1, 2]

        # Vectorized processing for speed
        v_coords, u_coords = np.meshgrid(range(self.depth_height), range(self.depth_width), indexing='ij')
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        depth_flat = depth_image.flatten()

        # Valid depth mask
        valid_mask = depth_flat > 0
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]

        # Convert depth pixel to 3D point in depth camera frame
        Z_d = depth_valid * depth_scale / 1000.0  # mm to meters
        X_d = (u_valid - cx_d) * Z_d / fx_d
        Y_d = (v_valid - cy_d) * Z_d / fy_d

        points_depth = np.stack([X_d, Y_d, Z_d], axis=1)  # (N, 3)

        # Transform to color camera frame: P_color = R * P_depth + t
        points_color = (self.R @ points_depth.T).T + self.t.T  # (N, 3)

        X_c = points_color[:, 0]
        Y_c = points_color[:, 1]
        Z_c = points_color[:, 2]

        # Filter points behind camera
        valid_proj = Z_c > 0
        X_c = X_c[valid_proj]
        Y_c = Y_c[valid_proj]
        Z_c = Z_c[valid_proj]
        depth_valid_proj = depth_valid[valid_proj]

        # Project to color camera image plane
        u_c = (fx_c * X_c / Z_c + cx_c).astype(int)
        v_c = (fy_c * Y_c / Z_c + cy_c).astype(int)

        # Filter points within image bounds
        valid_bounds = (u_c >= 0) & (u_c < self.rgb_width) & (v_c >= 0) & (v_c < self.rgb_height)
        u_c = u_c[valid_bounds]
        v_c = v_c[valid_bounds]
        depth_mm = depth_valid_proj[valid_bounds].astype(np.uint16)

        # Assign depth values (keep minimum depth per pixel to handle occlusions)
        for i in range(len(u_c)):
            u, v, d = u_c[i], v_c[i], depth_mm[i]
            if aligned_depth[v, u] == 0 or d < aligned_depth[v, u]:
                aligned_depth[v, u] = d

        return aligned_depth

    def align_with_hole_filling(self,
                                 depth_image: np.ndarray,
                                 depth_scale: float = 1.0,
                                 inpaint_radius: int = 3) -> np.ndarray:
        """
        Align depth with hole filling using inpainting

        Args:
            depth_image: Input depth image
            depth_scale: Depth scale factor
            inpaint_radius: Radius for inpainting

        Returns:
            Aligned depth image with holes filled
        """
        aligned_depth = self.align_depth_to_color(depth_image, depth_scale)

        # Fill holes using inpainting
        mask = (aligned_depth == 0).astype(np.uint8)
        if np.sum(mask) > 0:
            aligned_depth = cv2.inpaint(aligned_depth, mask, inpaint_radius, cv2.INPAINT_NS)

        return aligned_depth

    def visualize_alignment(self,
                           color_image: np.ndarray,
                           aligned_depth: np.ndarray,
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization overlay of color and aligned depth

        Args:
            color_image: RGB color image (3840×2160)
            aligned_depth: Aligned depth image (3840×2160)
            output_path: Optional path to save the visualization

        Returns:
            Visualization image
        """
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(aligned_depth, None, 0, 255,
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply colormap to depth
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Black out invalid depth
        depth_colored[aligned_depth == 0] = [0, 0, 0]

        # Ensure color image is in BGR format
        if len(color_image.shape) == 3 and color_image.shape[2] == 3:
            color_bgr = color_image
        else:
            color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Create overlay
        overlay = cv2.addWeighted(color_bgr, 0.6, depth_colored, 0.4, 0)

        # Add coverage info
        coverage = np.count_nonzero(aligned_depth) / aligned_depth.size * 100
        cv2.putText(overlay, f"Coverage: {coverage:.1f}%", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        if output_path:
            cv2.imwrite(output_path, overlay)
            logger.info(f"Saved visualization to: {output_path}")

        return overlay


def process_single_image(depth_path: str,
                         output_path: str,
                         aligner: DepthToColorAligner,
                         fill_holes: bool = False,
                         visualize: bool = False,
                         rgb_path: Optional[str] = None,
                         vis_output: Optional[str] = None) -> Dict:
    """
    Process a single depth image

    Args:
        depth_path: Path to depth image
        output_path: Output path for aligned depth
        aligner: DepthToColorAligner instance
        fill_holes: Whether to fill holes
        visualize: Whether to create visualization
        rgb_path: Optional RGB image path for visualization
        vis_output: Optional visualization output path

    Returns:
        Statistics dictionary
    """
    # Load depth image
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth image: {depth_path}")

    # Align
    if fill_holes:
        aligned = aligner.align_with_hole_filling(depth)
    else:
        aligned = aligner.align_depth_to_color(depth)

    # Save aligned depth
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, aligned)

    # Visualize if requested
    if visualize and rgb_path:
        rgb = cv2.imread(rgb_path)
        if rgb is not None:
            if vis_output is None:
                vis_output = str(Path(output_path).parent / f"{Path(output_path).stem}_vis.png")
            aligner.visualize_alignment(rgb, aligned, vis_output)

    # Statistics
    stats = {
        'input_shape': depth.shape,
        'output_shape': aligned.shape,
        'input_nonzero': int(np.count_nonzero(depth)),
        'output_nonzero': int(np.count_nonzero(aligned)),
        'coverage': float(np.count_nonzero(aligned) / aligned.size * 100)
    }

    return stats


def process_batch(depth_dir: str,
                  output_dir: str,
                  rgb_dir: Optional[str] = None,
                  fill_holes: bool = False,
                  visualize: bool = False,
                  limit: Optional[int] = None) -> None:
    """
    Process all depth images in a directory

    Args:
        depth_dir: Input depth directory
        output_dir: Output directory for aligned depth
        rgb_dir: Optional RGB directory for visualization
        fill_holes: Whether to fill holes
        visualize: Whether to create visualizations
        limit: Optional limit on number of images to process
    """
    logger.info("=" * 80)
    logger.info("Batch Depth-to-Color Alignment")
    logger.info("=" * 80)

    # Initialize aligner
    aligner = DepthToColorAligner()

    # Find depth images
    depth_path = Path(depth_dir)
    depth_files = sorted(depth_path.glob("*.png"))

    if limit:
        depth_files = depth_files[:limit]

    logger.info(f"\nFound {len(depth_files)} depth images")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if visualize:
        vis_dir = output_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

    # Process each image
    all_stats = []

    for depth_file in tqdm(depth_files, desc="Aligning depth images"):
        # Output path
        output_file = output_path / depth_file.name

        # Find matching RGB if available
        rgb_file = None
        if rgb_dir and visualize:
            rgb_path = Path(rgb_dir)
            # Try to find matching RGB (same stem)
            potential_rgb = rgb_path / depth_file.name
            if potential_rgb.exists():
                rgb_file = str(potential_rgb)

        vis_output = None
        if visualize:
            vis_output = str(vis_dir / f"{depth_file.stem}_overlay.png")

        try:
            stats = process_single_image(
                str(depth_file),
                str(output_file),
                aligner,
                fill_holes=fill_holes,
                visualize=visualize,
                rgb_path=rgb_file,
                vis_output=vis_output
            )
            all_stats.append(stats)

        except Exception as e:
            logger.error(f"Failed to process {depth_file.name}: {e}")
            continue

    # Summary statistics
    if all_stats:
        avg_coverage = np.mean([s['coverage'] for s in all_stats])
        logger.info("\n" + "=" * 80)
        logger.info("Summary")
        logger.info("=" * 80)
        logger.info(f"  Processed: {len(all_stats)} / {len(depth_files)} images")
        logger.info(f"  Average coverage: {avg_coverage:.1f}%")
        logger.info(f"  Output directory: {output_dir}")
        if visualize:
            logger.info(f"  Visualizations: {vis_dir}")


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(
        description='Align Depth to Color for Orbbec Femto Bolt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python -m src.align_depth_to_rgb --depth-image data/depth/sample.png --output aligned.png

  # Batch processing
  python -m src.align_depth_to_rgb --depth-dir data/depth --output-dir data/depth_upsampled

  # With visualization
  python -m src.align_depth_to_rgb --depth-dir data/depth --output-dir data/depth_upsampled \\
      --rgb-dir data/rgb --visualize
        """)

    # Input/output
    parser.add_argument('--depth-image', type=str,
                       help='Single depth image to process')
    parser.add_argument('--depth-dir', type=str,
                       help='Directory of depth images (batch processing)')
    parser.add_argument('--output', type=str,
                       help='Output path for single image')
    parser.add_argument('--output-dir', type=str, default='data/depth_upsampled',
                       help='Output directory for batch processing')

    # Optional
    parser.add_argument('--rgb-dir', type=str,
                       help='RGB directory for visualization (optional)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization overlays')
    parser.add_argument('--fill-holes', action='store_true',
                       help='Fill holes in aligned depth using inpainting')
    parser.add_argument('--limit', type=int,
                       help='Limit number of images to process (for testing)')

    # Calibration (use defaults)
    parser.add_argument('--rgb-calib', type=str, default='calib/rgb_camera_info.json',
                       help='RGB camera calibration JSON (default: calib/rgb_camera_info.json)')
    parser.add_argument('--depth-calib', type=str, default='calib/depth_camera_info.json',
                       help='Depth camera calibration JSON (default: calib/depth_camera_info.json)')
    parser.add_argument('--extrinsic', type=str, default='calib/extrinsic_depth_to_color.json',
                       help='Extrinsic calibration JSON (default: calib/extrinsic_depth_to_color.json)')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Validate input
    if not args.depth_image and not args.depth_dir:
        parser.error("Either --depth-image or --depth-dir must be specified")

    if args.depth_image and not args.output:
        parser.error("--output must be specified when using --depth-image")

    try:
        if args.depth_image:
            # Single image processing
            logger.info("Processing single image...")

            aligner = DepthToColorAligner(args.rgb_calib, args.depth_calib, args.extrinsic)

            rgb_path = None
            if args.visualize and args.rgb_dir:
                # Try to find matching RGB
                depth_stem = Path(args.depth_image).stem
                rgb_path = str(Path(args.rgb_dir) / f"{depth_stem}.png")
                if not Path(rgb_path).exists():
                    logger.warning(f"RGB image not found: {rgb_path}")
                    rgb_path = None

            stats = process_single_image(
                args.depth_image,
                args.output,
                aligner,
                fill_holes=args.fill_holes,
                visualize=args.visualize,
                rgb_path=rgb_path
            )

            logger.info(f"\n✓ Processing complete!")
            logger.info(f"  Coverage: {stats['coverage']:.1f}%")
            logger.info(f"  Output: {args.output}")

        else:
            # Batch processing
            process_batch(
                args.depth_dir,
                args.output_dir,
                rgb_dir=args.rgb_dir,
                fill_holes=args.fill_holes,
                visualize=args.visualize,
                limit=args.limit
            )

    except Exception as e:
        logger.error(f"Alignment failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
