"""
Plane-based Crack Detection Pipeline (Alternative to DBSCAN)

End-to-end pipeline:
1. Load YOLO detections for all images
2. Convert masks to 3D planes using depth + SFM
3. Cluster planes by coplanarity
4. Generate outputs and visualizations

Usage:
    python run_plane_pipeline.py \\
        --rgb-dir data/rgb \\
        --depth-dir data/depth \\
        --yolo-dir outputs/yolo_detections \\
        --colmap-model outputs/sfm/sparse/0 \\
        --output-dir outputs/plane_clustering \\
        --visualize
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import argparse

from src.mask_to_plane import MaskToPlaneConverter
from src.cluster_planes import PlaneClusterer

logger = logging.getLogger(__name__)


def find_matching_files(
    rgb_dir: Path,
    depth_dir: Path,
    yolo_dir: Path
) -> List[Dict[str, Path]]:
    """
    Find matching RGB, depth, and YOLO result files.

    Args:
        rgb_dir: Directory with RGB images
        depth_dir: Directory with aligned depth images
        yolo_dir: Directory with YOLO detection JSONs

    Returns:
        List of dicts with matching file paths
    """
    matches = []

    rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))

    for rgb_path in rgb_files:
        stem = rgb_path.stem

        # Find depth
        depth_path = depth_dir / f"{stem}.png"
        if not depth_path.exists():
            logger.warning(f"No depth image for {rgb_path.name}, skipping")
            continue

        # Find YOLO results
        yolo_path = yolo_dir / f"{stem}.json"
        if not yolo_path.exists():
            logger.warning(f"No YOLO results for {rgb_path.name}, skipping")
            continue

        matches.append({
            "rgb": rgb_path,
            "depth": depth_path,
            "yolo": yolo_path,
            "name": rgb_path.name
        })

    logger.info(f"Found {len(matches)} matching file sets")
    return matches


def visualize_planes(
    clusters: List[Dict],
    output_dir: Path
):
    """
    Create visualization of clustered planes.

    Generates:
    - 3D scatter plot of plane centers colored by cluster
    - Text summary of clusters

    Args:
        clusters: List of cluster summary dicts
        output_dir: Output directory
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
        return

    # Extract cluster centers
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))

    for i, cluster in enumerate(clusters):
        center = cluster['plane']['center']
        normal = cluster['plane']['normal']
        area = cluster['plane']['area_3d']

        # Plot center
        ax.scatter(
            center[0], center[1], center[2],
            color=colors[i],
            s=area * 1000,  # Size proportional to area
            alpha=0.6,
            label=f"Cluster {i+1} ({cluster['num_masks']} masks)"
        )

        # Plot normal vector
        ax.quiver(
            center[0], center[1], center[2],
            normal[0], normal[1], normal[2],
            length=0.2, color=colors[i], alpha=0.8
        )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Clustered Crack Planes')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    viz_path = output_dir / 'cluster_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {viz_path}")
    plt.close()


def generate_summary_report(
    clusters: List[Dict],
    output_dir: Path
):
    """
    Generate text summary report.

    Args:
        clusters: List of cluster summary dicts
        output_dir: Output directory
    """
    report_path = output_dir / 'summary.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PLANE-BASED CRACK CLUSTERING SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"Total crack area: {sum(c['plane']['area_3d'] for c in clusters):.2f} m²\n\n")

        for i, cluster in enumerate(clusters):
            f.write(f"\n{'='*80}\n")
            f.write(f"Cluster {i+1}\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Number of masks: {cluster['num_masks']}\n")
            f.write(f"  Number of views: {cluster['num_views']}\n")
            f.write(f"  Total area: {cluster['plane']['area_3d']:.2f} m²\n")
            f.write(f"  Confidence: mean={cluster['confidence']['mean']:.3f}, "
                   f"max={cluster['confidence']['max']:.3f}\n")
            f.write(f"\n  Plane parameters:\n")
            f.write(f"    Normal: [{cluster['plane']['normal'][0]:.3f}, "
                   f"{cluster['plane']['normal'][1]:.3f}, "
                   f"{cluster['plane']['normal'][2]:.3f}]\n")
            f.write(f"    d: {cluster['plane']['d']:.3f}\n")
            f.write(f"    Center: [{cluster['plane']['center'][0]:.3f}, "
                   f"{cluster['plane']['center'][1]:.3f}, "
                   f"{cluster['plane']['center'][2]:.3f}]\n")
            f.write(f"\n  Contributing images ({len(cluster['images'])}):\n")
            for img in cluster['images']:
                f.write(f"    - {img}\n")

        f.write("\n" + "="*80 + "\n")

    logger.info(f"Saved summary report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plane-based crack clustering pipeline"
    )

    # Input paths
    parser.add_argument(
        '--rgb-dir', type=Path, default=Path('data/rgb'),
        help='Directory with RGB images'
    )
    parser.add_argument(
        '--depth-dir', type=Path, default=Path('data/depth'),
        help='Directory with aligned depth images'
    )
    parser.add_argument(
        '--yolo-dir', type=Path, default=Path('outputs/yolo_detections'),
        help='Directory with YOLO detection JSONs'
    )
    parser.add_argument(
        '--colmap-model', type=Path, default=Path('outputs/sfm/sparse/0'),
        help='Path to COLMAP sparse model directory'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('outputs/plane_clustering'),
        help='Output directory for results'
    )

    # Calibration paths
    parser.add_argument(
        '--rgb-calib', type=Path, default=Path('calib/rgb_camera_info.json'),
        help='RGB camera calibration JSON'
    )
    parser.add_argument(
        '--depth-calib', type=Path, default=Path('calib/depth_camera_info.json'),
        help='Depth camera calibration JSON'
    )
    parser.add_argument(
        '--extrinsic', type=Path, default=Path('calib/extrinsic_depth_to_color.json'),
        help='Depth-to-RGB extrinsic JSON'
    )

    # Clustering parameters
    parser.add_argument(
        '--angle-threshold', type=float, default=15.0,
        help='Max angle between normals (degrees) [default: 15.0]'
    )
    parser.add_argument(
        '--distance-threshold', type=float, default=0.05,
        help='Max perpendicular distance between planes (meters) [default: 0.05]'
    )
    parser.add_argument(
        '--centroid-threshold', type=float, default=0.5,
        help='Max 3D distance between centroids (meters) [default: 0.5]'
    )
    parser.add_argument(
        '--min-planarity', type=float, default=0.7,
        help='Minimum planarity score [default: 0.7]'
    )
    parser.add_argument(
        '--min-coverage', type=float, default=0.3,
        help='Minimum depth coverage [default: 0.3]'
    )

    # Options
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    planes_dir = args.output_dir / 'planes'
    planes_dir.mkdir(exist_ok=True)

    print("="*80)
    print("PLANE-BASED CRACK CLUSTERING PIPELINE")
    print("="*80)

    # Step 1: Find matching files
    print("\n[1/3] Finding matching RGB-Depth-YOLO files...")
    file_matches = find_matching_files(args.rgb_dir, args.depth_dir, args.yolo_dir)

    if len(file_matches) == 0:
        logger.error("No matching files found!")
        return

    # Step 2: Convert masks to planes
    print("\n[2/3] Converting masks to 3D planes...")

    converter = MaskToPlaneConverter(
        rgb_calib_path=str(args.rgb_calib),
        depth_calib_path=str(args.depth_calib),
        extrinsic_path=str(args.extrinsic),
        colmap_model_path=str(args.colmap_model)
    )

    all_planes = []
    plane_files = []

    for match in tqdm(file_matches, desc="Processing images"):
        results = converter.process_image(
            image_path=str(match['rgb']),
            depth_path=str(match['depth']),
            yolo_results_path=str(match['yolo'])
        )

        if len(results) > 0:
            all_planes.extend(results)

            # Save per-image results
            output_path = planes_dir / f"{match['rgb'].stem}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            plane_files.append(output_path)

    logger.info(f"Converted {len(all_planes)} masks to planes")

    if len(all_planes) == 0:
        logger.error("No valid plane fits found!")
        return

    # Step 3: Cluster planes
    print("\n[3/3] Clustering planes...")

    clusterer = PlaneClusterer(
        angle_threshold=args.angle_threshold,
        distance_threshold=args.distance_threshold,
        centroid_threshold=args.centroid_threshold,
        min_planarity=args.min_planarity,
        min_depth_coverage=args.min_coverage
    )

    clusters = clusterer.cluster(all_planes)

    # Compute summaries
    cluster_summaries = [
        clusterer.compute_cluster_summary(cluster)
        for cluster in clusters
    ]

    # Save cluster results
    clusters_path = args.output_dir / 'clusters.json'
    with open(clusters_path, 'w') as f:
        json.dump(cluster_summaries, f, indent=2)

    logger.info(f"Saved {len(cluster_summaries)} clusters to {clusters_path}")

    # Generate outputs
    print("\nGenerating outputs...")
    generate_summary_report(cluster_summaries, args.output_dir)

    if args.visualize:
        visualize_planes(cluster_summaries, args.output_dir)

    # Final summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total planes extracted: {len(all_planes)}")
    print(f"Total clusters formed: {len(cluster_summaries)}")
    print(f"Total crack area: {sum(c['plane']['area_3d'] for c in cluster_summaries):.2f} m²")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("  - clusters.json: Cluster data")
    print("  - summary.txt: Human-readable summary")
    print("  - planes/*.json: Per-image plane fits")
    if args.visualize:
        print("  - cluster_visualization.png: 3D scatter plot")
    print("="*80)


if __name__ == '__main__':
    main()
