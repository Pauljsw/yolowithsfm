"""
Lightweight Mask-level 3D Grouping

Groups masks based on 3D centroid proximity (no DBSCAN, no plane fitting).

Grouping criteria:
1. Centroid distance < threshold
2. (Optional) Depth similarity
3. (Optional) Size similarity

Benefits:
- No density dependence
- Fast (O(n²) distance computation)
- Intuitive (spatial proximity)
- Works with sparse or dense data
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Set
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class MaskGrouper:
    """Group masks based on 3D centroid proximity"""

    def __init__(
        self,
        distance_threshold: float = 0.5,     # meters
        depth_diff_threshold: float = 0.2,   # meters
        min_confidence: float = 0.25,
        use_depth_similarity: bool = True
    ):
        """
        Args:
            distance_threshold: Max 3D distance between centroids (meters)
            depth_diff_threshold: Max depth difference to group (meters)
            min_confidence: Minimum YOLO confidence to include
            use_depth_similarity: Whether to use depth as grouping criterion
        """
        self.distance_threshold = distance_threshold
        self.depth_diff_threshold = depth_diff_threshold
        self.min_confidence = min_confidence
        self.use_depth_similarity = use_depth_similarity

    def should_group(self, mask1: Dict, mask2: Dict) -> bool:
        """
        Check if two masks should be grouped.

        Args:
            mask1, mask2: Mask 3D data dicts

        Returns:
            True if should be grouped
        """
        centroid1 = np.array(mask1['centroid_3d'])
        centroid2 = np.array(mask2['centroid_3d'])

        # Check 1: Centroid distance
        distance = np.linalg.norm(centroid1 - centroid2)
        if distance > self.distance_threshold:
            return False

        # Check 2: Depth similarity (optional)
        if self.use_depth_similarity:
            depth_diff = abs(mask1['depth_median'] - mask2['depth_median'])
            if depth_diff > self.depth_diff_threshold:
                return False

        return True

    def filter_quality(self, masks: List[Dict]) -> List[Dict]:
        """
        Filter masks by confidence threshold.

        Args:
            masks: List of mask 3D dicts

        Returns:
            Filtered list
        """
        filtered = []
        for mask in masks:
            if mask['confidence'] >= self.min_confidence:
                filtered.append(mask)
            else:
                logger.debug(
                    f"[{mask['image']}] Mask {mask['mask_idx']}: "
                    f"Filtered out (confidence={mask['confidence']:.3f})"
                )

        logger.info(
            f"Quality filter: {len(filtered)}/{len(masks)} masks passed "
            f"(confidence≥{self.min_confidence})"
        )
        return filtered

    def group(self, masks: List[Dict]) -> List[List[Dict]]:
        """
        Group masks by 3D centroid proximity.

        Uses greedy agglomerative clustering.

        Args:
            masks: List of mask 3D dicts

        Returns:
            List of groups, each group is a list of mask dicts
        """
        # Filter by quality
        masks = self.filter_quality(masks)

        if len(masks) == 0:
            return []

        # Initialize: each mask is its own group
        groups = [[mask] for mask in masks]

        # Greedy merging
        merged = True
        iteration = 0

        while merged and len(groups) > 1:
            merged = False
            iteration += 1

            # Try to merge groups
            i = 0
            while i < len(groups):
                j = i + 1
                while j < len(groups):
                    # Check if groups should merge
                    # (any pair of masks from different groups is close enough)
                    should_merge = False

                    for mask1 in groups[i]:
                        for mask2 in groups[j]:
                            if self.should_group(mask1, mask2):
                                should_merge = True
                                break
                        if should_merge:
                            break

                    if should_merge:
                        # Merge group j into group i
                        groups[i].extend(groups[j])
                        del groups[j]
                        merged = True
                        logger.debug(
                            f"Iteration {iteration}: Merged groups "
                            f"(now {len(groups)} groups)"
                        )
                    else:
                        j += 1

                i += 1

        logger.info(f"Grouping complete: {len(groups)} groups from {len(masks)} masks")

        # Sort groups by size (largest first)
        groups.sort(key=lambda g: len(g), reverse=True)

        return groups

    def compute_group_summary(self, group: List[Dict]) -> Dict:
        """
        Compute summary statistics for a group.

        Args:
            group: List of mask dicts in the group

        Returns:
            Summary dict
        """
        # Average centroid (unweighted for simplicity)
        centroids = np.array([m['centroid_3d'] for m in group])
        avg_centroid = centroids.mean(axis=0)

        # Average depth
        depths = [m['depth_median'] for m in group]
        avg_depth = np.mean(depths)

        # Total area (2D, for now)
        total_area_2d = sum(m['mask_area_2d'] for m in group)

        # Estimate 3D extent
        # Compute bounding box of all centroids
        min_coords = centroids.min(axis=0)
        max_coords = centroids.max(axis=0)
        extent_3d = (max_coords - min_coords).tolist()

        # Confidence statistics
        confidences = [m['confidence'] for m in group]
        max_confidence = max(confidences)
        avg_confidence = np.mean(confidences)

        # Contributing images
        images = list(set(m['image'] for m in group))

        return {
            "num_masks": len(group),
            "num_views": len(images),
            "centroid_3d": avg_centroid.tolist(),
            "depth_median": float(avg_depth),
            "extent_3d": extent_3d,  # [dx, dy, dz] in meters
            "area_2d_total": int(total_area_2d),  # Total pixels
            "confidence": {
                "max": float(max_confidence),
                "mean": float(avg_confidence)
            },
            "images": images,
            "masks": [
                {
                    "image": m['image'],
                    "mask_idx": m['mask_idx'],
                    "confidence": m['confidence'],
                    "centroid_3d": m['centroid_3d']
                }
                for m in group
            ]
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Group masks by 3D centroid proximity"
    )
    parser.add_argument(
        '--input', required=True, nargs='+',
        help='Input JSON files with mask 3D data'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output JSON path for groups'
    )
    parser.add_argument(
        '--distance-threshold', type=float, default=0.5,
        help='Max 3D distance between centroids (meters)'
    )
    parser.add_argument(
        '--depth-diff-threshold', type=float, default=0.2,
        help='Max depth difference (meters)'
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.25,
        help='Minimum YOLO confidence'
    )
    parser.add_argument(
        '--no-depth-similarity', action='store_true',
        help='Disable depth similarity check'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Load all mask 3D data
    all_masks = []
    for input_path in args.input:
        with open(input_path) as f:
            masks = json.load(f)
            all_masks.extend(masks)
            logger.info(f"Loaded {len(masks)} masks from {input_path}")

    logger.info(f"Total masks: {len(all_masks)}")

    # Group
    grouper = MaskGrouper(
        distance_threshold=args.distance_threshold,
        depth_diff_threshold=args.depth_diff_threshold,
        min_confidence=args.min_confidence,
        use_depth_similarity=not args.no_depth_similarity
    )

    groups = grouper.group(all_masks)

    # Compute summaries
    group_summaries = [
        grouper.compute_group_summary(group)
        for group in groups
    ]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(group_summaries, f, indent=2)

    logger.info(f"Saved {len(group_summaries)} groups to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("GROUPING SUMMARY")
    print("="*80)
    for i, summary in enumerate(group_summaries):
        print(f"\nGroup {i+1}:")
        print(f"  Masks: {summary['num_masks']}")
        print(f"  Views: {summary['num_views']}")
        print(f"  Extent: {summary['extent_3d']} m")
        print(f"  Confidence: {summary['confidence']['mean']:.3f} "
              f"(max: {summary['confidence']['max']:.3f})")
        print(f"  Centroid: {summary['centroid_3d']}")
    print("="*80)


if __name__ == '__main__':
    main()
