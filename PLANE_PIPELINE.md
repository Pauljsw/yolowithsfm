# Plane-Based Crack Detection Pipeline

## Overview

This is an alternative to the DBSCAN-based clustering approach that leverages the **planar crack assumption** for more robust and efficient grouping.

### Key Idea
Most cracks occur on **planar surfaces** (walls, floors, ceilings). Instead of clustering sparse 3D point clouds with DBSCAN (density-dependent), we:
1. Fit a plane to each crack mask's 3D point cloud
2. Cluster cracks by comparing plane parameters (coplanarity)

### Benefits over DBSCAN
- ✅ **Not density-dependent**: Works regardless of point cloud sparsity
- ✅ **Semantically meaningful**: Groups cracks on the same surface
- ✅ **Efficient**: Compares 4 plane parameters instead of thousands of points
- ✅ **Robust to occlusion**: Handles partial observations well
- ✅ **Leverages RGB-D data**: Uses depth for accurate 3D reconstruction

---

## Architecture

### Phase 3 Alternative: Mask → Plane Transformation
**Module**: `src/mask_to_plane.py`

**Input**:
- RGB image with YOLO mask detections
- Aligned depth image (512×512)
- COLMAP camera pose (from `images.bin`)
- Camera calibration (RGB + depth intrinsics)

**Process**:
1. For each YOLO mask polygon:
   - Get mask pixel coordinates in RGB frame
   - Scale to depth image coordinates (512×512)
   - Lookup depth values (in mm)
   - Backproject to 3D camera frame using RGB intrinsics
   - Transform to global 3D using COLMAP pose
2. Fit plane to 3D points using PCA:
   - Plane normal = eigenvector with smallest eigenvalue
   - Plane equation: `n·P + d = 0`
   - Compute planarity score (fit quality)

**Output** (per mask):
```json
{
  "image": "IMG_001.jpg",
  "mask_idx": 0,
  "confidence": 0.85,
  "plane": {
    "normal": [0.02, -0.99, 0.12],  // Unit normal vector
    "d": -2.45,                      // Distance from origin
    "center": [1.2, 2.3, -2.4],      // 3D centroid
    "area_3d": 0.15,                 // Surface area (m²)
    "num_points": 4823               // Valid 3D points
  },
  "quality": {
    "planarity": 0.95,               // How well points fit plane (0-1)
    "depth_coverage": 0.78           // % of mask with valid depth
  }
}
```

### Phase 4 Replacement: Plane-Based Clustering
**Module**: `src/cluster_planes.py`

**Input**:
- List of mask plane data (from all images)

**Grouping Criteria** (all must be satisfied):
1. **Normal similarity**: `angle(n1, n2) < 15°`
2. **Distance similarity**: `|d1 - d2| < 0.05 m`
3. **Centroid proximity**: `||center1 - center2|| < 0.5 m`

**Algorithm**:
- Greedy agglomerative clustering
- Quality filtering (planarity ≥ 0.7, coverage ≥ 0.3)
- Merge clusters if any pair of masks is coplanar

**Output** (per cluster):
```json
{
  "num_masks": 5,
  "num_views": 3,
  "plane": {
    "normal": [0.02, -0.99, 0.12],  // Weighted average
    "d": -2.45,
    "center": [1.2, 2.3, -2.4],
    "area_3d": 0.85                  // Total area
  },
  "confidence": {
    "max": 0.92,
    "mean": 0.78
  },
  "images": ["IMG_001.jpg", "IMG_002.jpg", "IMG_005.jpg"],
  "masks": [...]
}
```

---

## Usage

### Prerequisites
1. **YOLO detections** for all images (JSON format)
2. **Aligned depth images** (512×512, PNG, 16-bit, mm units)
3. **COLMAP sparse reconstruction** (`images.bin`, `cameras.bin`, `points3D.bin`)
4. **Camera calibration files**:
   - `calib/rgb_camera_info.json`
   - `calib/depth_camera_info.json`
   - `calib/extrinsic_depth_to_color.json`

### Run Full Pipeline

```bash
python run_plane_pipeline.py \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --yolo-dir outputs/yolo_detections \
    --colmap-model outputs/sfm/sparse/0 \
    --output-dir outputs/plane_clustering \
    --visualize
```

### Advanced Options

**Clustering thresholds**:
```bash
--angle-threshold 15.0          # Max angle between normals (degrees)
--distance-threshold 0.05       # Max perpendicular distance (meters)
--centroid-threshold 0.5        # Max centroid distance (meters)
```

**Quality filters**:
```bash
--min-planarity 0.7             # Minimum planarity score
--min-coverage 0.3              # Minimum depth coverage
```

### Step-by-Step (Manual)

**Step 1**: Convert single image masks to planes
```bash
python -m src.mask_to_plane \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --extrinsic calib/extrinsic_depth_to_color.json \
    --colmap-model outputs/sfm/sparse/0 \
    --image data/rgb/IMG_001.jpg \
    --depth data/depth/IMG_001.png \
    --yolo-results outputs/yolo_detections/IMG_001.json \
    --output outputs/planes/IMG_001.json
```

**Step 2**: Cluster planes
```bash
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 15.0 \
    --distance-threshold 0.05 \
    --centroid-threshold 0.5
```

---

## Output Files

After running the pipeline, you'll find:

```
outputs/plane_clustering/
├── clusters.json                   # Cluster summaries (JSON)
├── summary.txt                     # Human-readable report
├── cluster_visualization.png       # 3D scatter plot (if --visualize)
└── planes/
    ├── IMG_001.json                # Per-image plane fits
    ├── IMG_002.json
    └── ...
```

### Summary Report Example
```
================================================================================
PLANE-BASED CRACK CLUSTERING SUMMARY
================================================================================

Total clusters: 8
Total crack area: 2.45 m²

================================================================================
Cluster 1
================================================================================
  Number of masks: 12
  Number of views: 5
  Total area: 0.85 m²
  Confidence: mean=0.78, max=0.92

  Plane parameters:
    Normal: [0.02, -0.99, 0.12]
    d: -2.45
    Center: [1.2, 2.3, -2.4]

  Contributing images (5):
    - IMG_001.jpg
    - IMG_002.jpg
    - IMG_005.jpg
    - IMG_008.jpg
    - IMG_012.jpg
```

---

## Comparison: DBSCAN vs Plane-Based

| Aspect | DBSCAN (Old) | Plane-Based (New) |
|--------|--------------|-------------------|
| **Input** | Sparse track points | Dense depth-based point cloud |
| **Clustering** | Density-based | Geometric (coplanarity) |
| **Assumption** | None (generic) | Planar surfaces |
| **Density dependence** | ❌ Yes (eps tuning nightmare) | ✅ No |
| **Coverage** | ~50-200 points/crack | ~5000 points/crack |
| **Efficiency** | O(n²) for n points | O(m²) for m masks |
| **Interpretability** | Low (abstract clusters) | High (plane = surface) |
| **Robustness** | Sensitive to sparsity | Robust to occlusion |

---

## Key Design Decisions

### Why Planar Assumption?
- **Empirical validity**: Most infrastructure cracks occur on walls, floors, ceilings (planar)
- **Simplifies representation**: 4 parameters vs thousands of points
- **Enables semantic grouping**: "Cracks on the same wall" is meaningful
- **Graceful degradation**: Non-planar cracks still work (lower planarity score)

### Why Use Depth Instead of SFM Tracks?
- **Coverage**: SFM tracks are sparse (50-200 points/mask), depth is dense (~5000 points/mask)
- **Accuracy**: Direct depth measurement vs triangulation error accumulation
- **Independence**: Depth works even for untextured regions where SFM fails
- **Availability**: RGB-D sensor provides depth "for free"

### Quality Filtering
Two-stage filtering ensures robust results:
1. **Per-mask**: Planarity ≥ 0.7, coverage ≥ 0.3
2. **Per-cluster**: Minimum 2 views (reject spurious detections)

---

## Limitations and Extensions

### Current Limitations
1. **Planar assumption**: May fail for cracks on curved surfaces (pipes, arches)
2. **Depth dependency**: Requires aligned depth (no depth = no plane)
3. **Occlusion handling**: Partial masks may have biased plane fits

### Possible Extensions
1. **Curved surface support**: Use quadric fitting instead of planes
2. **Hybrid approach**: Combine with sparse track points when depth unavailable
3. **Temporal consistency**: Track planes across video sequences
4. **Plane-aware SLAM**: Use detected planes to improve COLMAP reconstruction

---

## Troubleshooting

### "No valid plane fits found"
- Check depth images are aligned (not raw sensor coordinates)
- Verify depth scale is correct (default: 1000 = mm → m)
- Lower quality thresholds (`--min-planarity 0.5 --min-coverage 0.2`)

### "Too many clusters" (over-segmentation)
- Increase thresholds:
  ```bash
  --angle-threshold 20.0
  --distance-threshold 0.10
  --centroid-threshold 1.0
  ```

### "Too few clusters" (under-segmentation)
- Decrease thresholds:
  ```bash
  --angle-threshold 10.0
  --distance-threshold 0.03
  --centroid-threshold 0.3
  ```

### "Low planarity scores"
- Increase `--min-planarity` to filter noisy detections
- Check depth image quality (noise, missing data)
- Inspect YOLO masks (incorrect segmentation?)

---

## References

- **PCA-based plane fitting**: Pearson (1901), "On lines and planes of closest fit"
- **Coplanarity metrics**: Fischler & Bolles (1981), RANSAC
- **RGB-D alignment**: Orbbec SDK documentation
- **COLMAP conventions**: Schönberger & Frahm (2016)
