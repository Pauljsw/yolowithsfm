"""
Plane-based Clustering (Phase 4 Replacement)

Groups masks by comparing their fitted plane parameters instead of using
density-based DBSCAN on point clouds.

Grouping criteria:
1. Normal similarity: angle between plane normals < threshold
2. Distance similarity: perpendicular distance between planes < threshold
3. Centroid proximity: 3D distance between centroids < threshold

Benefits over DBSCAN:
- Not density-dependent (works with any point cloud density)
- Semantically meaningful (groups coplanar cracks)
- Efficient (compares 4 parameters instead of thousands of points)
- Robust to occlusion and sparse data
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class PlaneClusterer:
    """Cluster masks based on their plane parameters"""

    def __init__(
        self,
        angle_threshold: float = 15.0,      # degrees
        distance_threshold: float = 0.05,   # meters
        centroid_threshold: float = 0.5,    # meters
        min_planarity: float = 0.7,         # quality filter
        min_depth_coverage: float = 0.3     # quality filter
    ):
        """
        Args:
            angle_threshold: Max angle between normals (degrees) to be coplanar
            distance_threshold: Max perpendicular distance between planes (meters)
            centroid_threshold: Max 3D distance between centroids (meters)
            min_planarity: Minimum planarity score to include mask
            min_depth_coverage: Minimum depth coverage to include mask
        """
        self.angle_threshold = np.deg2rad(angle_threshold)
        self.distance_threshold = distance_threshold
        self.centroid_threshold = centroid_threshold
        self.min_planarity = min_planarity
        self.min_depth_coverage = min_depth_coverage

    def compute_plane_distance(
        self,
        normal1: np.ndarray,
        d1: float,
        normal2: np.ndarray,
        d2: float
    ) -> float:
        """
        Compute perpendicular distance between two parallel planes.

        For planes: n1·P + d1 = 0 and n2·P + d2 = 0
        If normals are similar, distance ≈ |d1 - d2| / ||n||

        Args:
            normal1: (3,) unit normal of plane 1
            d1: Distance parameter of plane 1
            normal2: (3,) unit normal of plane 2
            d2: Distance parameter of plane 2

        Returns:
            Perpendicular distance between planes (meters)
        """
        # If normals point in opposite directions, flip one
        if np.dot(normal1, normal2) < 0:
            normal2 = -normal2
            d2 = -d2

        # Distance between parallel planes
        return abs(d1 - d2)

    def compute_normal_angle(
        self,
        normal1: np.ndarray,
        normal2: np.ndarray
    ) -> float:
        """
        Compute angle between two plane normals.

        Args:
            normal1: (3,) unit normal
            normal2: (3,) unit normal

        Returns:
            Angle in radians [0, π/2]
        """
        dot_product = np.clip(np.abs(np.dot(normal1, normal2)), -1.0, 1.0)
        return np.arccos(dot_product)

    def are_coplanar(
        self,
        mask1: Dict,
        mask2: Dict
    ) -> bool:
        """
        Check if two masks lie on the same plane.

        Args:
            mask1: Mask plane data dict
            mask2: Mask plane data dict

        Returns:
            True if coplanar, False otherwise
        """
        plane1 = mask1['plane']
        plane2 = mask2['plane']

        normal1 = np.array(plane1['normal'])
        normal2 = np.array(plane2['normal'])
        d1 = plane1['d']
        d2 = plane2['d']
        center1 = np.array(plane1['center'])
        center2 = np.array(plane2['center'])

        # Check 1: Normal similarity
        angle = self.compute_normal_angle(normal1, normal2)
        if angle > self.angle_threshold:
            return False

        # Check 2: Plane distance
        plane_dist = self.compute_plane_distance(normal1, d1, normal2, d2)
        if plane_dist > self.distance_threshold:
            return False

        # Check 3: Centroid proximity
        centroid_dist = np.linalg.norm(center1 - center2)
        if centroid_dist > self.centroid_threshold:
            return False

        return True

    def filter_quality(self, masks: List[Dict]) -> List[Dict]:
        """
        Filter masks by quality thresholds.

        Args:
            masks: List of mask plane dicts

        Returns:
            Filtered list of high-quality masks
        """
        filtered = []
        for mask in masks:
            quality = mask['quality']
            if (quality['planarity'] >= self.min_planarity and
                quality['depth_coverage'] >= self.min_depth_coverage):
                filtered.append(mask)
            else:
                logger.debug(
                    f"[{mask['image']}] Mask {mask['mask_idx']}: "
                    f"Filtered out (planarity={quality['planarity']:.3f}, "
                    f"coverage={quality['depth_coverage']:.3f})"
                )

        logger.info(
            f"Quality filter: {len(filtered)}/{len(masks)} masks passed "
            f"(planarity≥{self.min_planarity}, coverage≥{self.min_depth_coverage})"
        )
        return filtered

    def cluster(self, masks: List[Dict]) -> List[List[Dict]]:
        """
        Cluster masks into coplanar groups.

        Uses greedy agglomerative clustering based on plane similarity.

        Args:
            masks: List of mask plane dicts

        Returns:
            List of clusters, each cluster is a list of mask dicts
        """
        # Filter by quality
        masks = self.filter_quality(masks)

        if len(masks) == 0:
            return []

        # Initialize: each mask is its own cluster
        clusters = [[mask] for mask in masks]
        mask_to_cluster = {id(mask): i for i, mask in enumerate(masks)}

        # Greedy merging
        merged = True
        iteration = 0

        while merged and len(clusters) > 1:
            merged = False
            iteration += 1

            # Try to merge clusters
            i = 0
            while i < len(clusters):
                j = i + 1
                while j < len(clusters):
                    # Check if clusters should merge
                    # (any pair of masks from different clusters is coplanar)
                    should_merge = False

                    for mask1 in clusters[i]:
                        for mask2 in clusters[j]:
                            if self.are_coplanar(mask1, mask2):
                                should_merge = True
                                break
                        if should_merge:
                            break

                    if should_merge:
                        # Merge cluster j into cluster i
                        clusters[i].extend(clusters[j])
                        del clusters[j]
                        merged = True
                        logger.debug(
                            f"Iteration {iteration}: Merged clusters "
                            f"(now {len(clusters)} clusters)"
                        )
                    else:
                        j += 1

                i += 1

        logger.info(f"Clustering complete: {len(clusters)} clusters from {len(masks)} masks")

        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: len(c), reverse=True)

        return clusters

    def compute_cluster_summary(self, cluster: List[Dict]) -> Dict:
        """
        Compute summary statistics for a cluster.

        Args:
            cluster: List of mask dicts in the cluster

        Returns:
            Summary dict with merged plane, statistics, etc.
        """
        # Compute merged plane (weighted average by num_points)
        total_points = sum(m['plane']['num_points'] for m in cluster)
        weights = np.array([m['plane']['num_points'] / total_points for m in cluster])

        # Average normal (weighted)
        normals = np.array([m['plane']['normal'] for m in cluster])
        avg_normal = (normals * weights[:, None]).sum(axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)  # Normalize

        # Average d (weighted)
        ds = np.array([m['plane']['d'] for m in cluster])
        avg_d = (ds * weights).sum()

        # Average center (weighted)
        centers = np.array([m['plane']['center'] for m in cluster])
        avg_center = (centers * weights[:, None]).sum(axis=0)

        # Total area
        total_area = sum(m['plane']['area_3d'] for m in cluster)

        # Confidence statistics
        confidences = [m['confidence'] for m in cluster]
        max_confidence = max(confidences)
        avg_confidence = np.mean(confidences)

        # Contributing images
        images = list(set(m['image'] for m in cluster))

        return {
            "num_masks": len(cluster),
            "plane": {
                "normal": avg_normal.tolist(),
                "d": float(avg_d),
                "center": avg_center.tolist(),
                "area_3d": float(total_area)
            },
            "confidence": {
                "max": float(max_confidence),
                "mean": float(avg_confidence)
            },
            "images": images,
            "num_views": len(images),
            "masks": [
                {
                    "image": m['image'],
                    "mask_idx": m['mask_idx'],
                    "confidence": m['confidence']
                }
                for m in cluster
            ]
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster masks by plane similarity (replaces DBSCAN)"
    )
    parser.add_argument(
        '--input', required=True, nargs='+',
        help='Input JSON files with mask plane data'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output JSON path for clusters'
    )
    parser.add_argument(
        '--angle-threshold', type=float, default=15.0,
        help='Max angle between normals (degrees)'
    )
    parser.add_argument(
        '--distance-threshold', type=float, default=0.05,
        help='Max perpendicular distance between planes (meters)'
    )
    parser.add_argument(
        '--centroid-threshold', type=float, default=0.5,
        help='Max 3D distance between centroids (meters)'
    )
    parser.add_argument(
        '--min-planarity', type=float, default=0.7,
        help='Minimum planarity score'
    )
    parser.add_argument(
        '--min-coverage', type=float, default=0.3,
        help='Minimum depth coverage'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Load all mask plane data
    all_masks = []
    for input_path in args.input:
        with open(input_path) as f:
            masks = json.load(f)
            all_masks.extend(masks)
            logger.info(f"Loaded {len(masks)} masks from {input_path}")

    logger.info(f"Total masks: {len(all_masks)}")

    # Cluster
    clusterer = PlaneClusterer(
        angle_threshold=args.angle_threshold,
        distance_threshold=args.distance_threshold,
        centroid_threshold=args.centroid_threshold,
        min_planarity=args.min_planarity,
        min_depth_coverage=args.min_coverage
    )

    clusters = clusterer.cluster(all_masks)

    # Compute summaries
    cluster_summaries = [
        clusterer.compute_cluster_summary(cluster)
        for cluster in clusters
    ]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(cluster_summaries, f, indent=2)

    logger.info(f"Saved {len(cluster_summaries)} clusters to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("CLUSTERING SUMMARY")
    print("="*80)
    for i, summary in enumerate(cluster_summaries):
        print(f"\nCluster {i+1}:")
        print(f"  Masks: {summary['num_masks']}")
        print(f"  Views: {summary['num_views']}")
        print(f"  Area: {summary['plane']['area_3d']:.2f} m²")
        print(f"  Confidence: {summary['confidence']['mean']:.3f} "
              f"(max: {summary['confidence']['max']:.3f})")
        print(f"  Normal: [{summary['plane']['normal'][0]:.3f}, "
              f"{summary['plane']['normal'][1]:.3f}, "
              f"{summary['plane']['normal'][2]:.3f}]")
    print("="*80)


if __name__ == '__main__':
    main()
