# Complete Execution Guide: Lightweight Mask-based Pipeline

## Overview

This pipeline converts YOLO crack detection masks to 3D, groups them by spatial proximity, and visualizes them in the SFM global coordinate system.

**Key Features**:
- ✅ Lightweight: Only computes mask centroids (not full point clouds)
- ✅ Fast: Simple distance-based grouping (no DBSCAN)
- ✅ Clear visualization: SFM point cloud + colored crack groups

---

## Prerequisites

Ensure you have completed:

```bash
# 1. COLMAP SFM reconstruction
python -m src.pipeline sfm --config configs/simple.yaml

# 2. YOLO crack detection
python -m src.pipeline detect --config configs/simple.yaml
```

**Verify**:
```bash
ls data/sfm/sparse/0/images.bin     # ✅ Must exist
ls data/yolo_masks/*.json | wc -l   # ✅ Should show number of detections
ls data/depth/*.png | wc -l         # ✅ Should match RGB image count
```

---

## Phase 3: Mask → 3D Conversion (Batch Mode)

Convert all YOLO masks to lightweight 3D representations using depth + COLMAP poses.

### Command

```bash
python -m src.mask_to_3d \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --yolo-dir data/yolo_masks \
    --output-dir outputs/masks_3d
```

### What it Does

For each mask in each image:
1. Loads COLMAP camera pose (once for all images - fast!)
2. Computes mask centroid in 2D
3. Gets median depth value within mask
4. Backprojects centroid to 3D using depth + camera intrinsics
5. Transforms to SFM global coordinates

### Output Format

Each `outputs/masks_3d/camera_RGB_<timestamp>.json`:
```json
[
  {
    "image": "camera_RGB_1758853283_533442048.png",
    "mask_idx": 0,
    "confidence": 0.85,
    "centroid_2d": [1920, 1080],
    "centroid_3d": [1.234, 2.345, -2.456],
    "depth_median": 2.45,
    "depth_std": 0.12,
    "mask_area_2d": 15000,
    "bbox_2d": [1800, 1000, 2040, 1160],
    "bbox_3d_size": [0.15, 0.15, 0.24],
    "depth_coverage": 0.78
  }
]
```

### Verify

```bash
# Check number of processed files
ls outputs/masks_3d/*.json | wc -l

# Check total masks
python -c "
import json
from pathlib import Path

total = 0
for f in Path('outputs/masks_3d').glob('*.json'):
    with open(f) as fp:
        total += len(json.load(fp))
print(f'Total masks with 3D data: {total}')
"

# Inspect one file
cat outputs/masks_3d/camera_RGB_*.json | python -m json.tool | head -30
```

---

## Phase 4: Mask Grouping

Group masks by 3D centroid proximity.

### Command (Default Parameters)

```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --depth-diff-threshold 0.2 \
    --min-confidence 0.25
```

### Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--distance-threshold` | 0.5 | Maximum 3D distance (meters) between mask centroids to group together |
| `--depth-diff-threshold` | 0.2 | Maximum depth difference (meters) to consider as same crack |
| `--min-confidence` | 0.25 | Minimum YOLO confidence to include mask |
| `--no-depth-similarity` | - | Disable depth check (only use centroid distance) |

### Grouping Logic

Two masks are grouped if:
1. **Centroid distance** < `distance_threshold` (in 3D space)
2. **Depth similarity** (optional): |depth₁ - depth₂| < `depth_diff_threshold`
3. Both have confidence ≥ `min_confidence`

Uses greedy agglomerative clustering (NOT DBSCAN - density-independent!).

### Output Format

`outputs/groups.json`:
```json
[
  {
    "num_masks": 12,
    "num_views": 5,
    "centroid_3d": [1.234, 2.345, -2.456],
    "depth_median": 2.45,
    "extent_3d": [0.30, 0.40, 0.10],
    "area_2d_total": 180000,
    "confidence": {
      "max": 0.92,
      "mean": 0.78
    },
    "images": ["camera_RGB_001.png", "camera_RGB_002.png", ...],
    "masks": [
      {
        "image": "camera_RGB_001.png",
        "mask_idx": 0,
        "centroid_3d": [1.2, 2.3, -2.4],
        ...
      }
    ]
  }
]
```

**Fields**:
- `num_masks`: Total number of masks in group
- `num_views`: Number of unique images (views) containing this group
- `centroid_3d`: Group center in SFM global coordinates
- `extent_3d`: [dx, dy, dz] bounding box size in meters
- `depth_median`: Median depth of all masks in group
- `area_2d_total`: Sum of mask pixel areas

### Verify

```bash
# Check grouping results
python -c "
import json

groups = json.load(open('outputs/groups.json'))
print(f'Total groups: {len(groups)}\n')

for i, g in enumerate(groups[:10]):  # Show first 10
    print(f'Group {i+1}:')
    print(f'  Masks: {g[\"num_masks\"]}, Views: {g[\"num_views\"]}')
    print(f'  Centroid: {g[\"centroid_3d\"]}')
    print(f'  Extent: {g[\"extent_3d\"]} m')
    print(f'  Confidence: {g[\"confidence\"][\"mean\"]:.3f}')
    print()
"
```

### Tuning Parameters

**If groups are over-fragmented** (too many small groups):
```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 1.0 \      # Increase from 0.5
    --depth-diff-threshold 0.5 \    # Increase from 0.2
    --min-confidence 0.25
```

**If groups are over-merged** (few large groups):
```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.3 \      # Decrease from 0.5
    --depth-diff-threshold 0.1 \    # Decrease from 0.2
    --min-confidence 0.25
```

**Ignore depth similarity** (only use spatial distance):
```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --min-confidence 0.25 \
    --no-depth-similarity           # Disable depth check
```

---

## Phase 5: Visualization

### Option 1: Open3D Interactive Viewer (Recommended)

**Best for**: Quick exploration, reliable rendering, no external tools needed

```bash
python visualize_simple.py \
    --groups outputs/groups.json \
    --masks-3d-dir outputs/masks_3d \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --min-views 2 \
    --sample-rate 10 \
    --max-sparse-points 50000
```

**Parameters**:
- `--min-views 2`: Only show groups seen in ≥2 views (filters noise)
- `--sample-rate 10`: Sample every 10th pixel (lower = denser but slower)
- `--max-sparse-points 50000`: Max COLMAP points to show as background
- `--save output.ply`: Optional - save to PLY file

**What you'll see**:
- Gray point cloud (COLMAP sparse reconstruction - background reference)
- Colored regions (each crack group in different color)
- Interactive 3D viewer with rotation, zoom, pan

**Controls**:
- Left mouse: Rotate
- Right mouse: Pan
- Scroll: Zoom
- ESC: Close

---

### Option 2: Matplotlib Static Plot

**Best for**: Saving publication-quality images

```bash
python -m src.visualize_groups_3d \
    --colmap-model data/sfm/sparse/0 \
    --groups outputs/groups.json \
    --output outputs/visualization_3d.png \
    --max-points 50000 \
    --export-ply outputs/groups_with_sparse.ply
```

**Parameters**:
- `--output`: Save to image file
- `--show`: Show interactive matplotlib window instead
- `--no-extent`: Don't show bounding boxes
- `--export-ply`: Export to PLY for CloudCompare

---

### Option 3: Mesh Export for CloudCompare

Export groups as 3D meshes (surfaces, not points).

**Simple bounding boxes** (fast):
```bash
python -m src.export_group_meshes \
    --groups outputs/groups.json \
    --masks-3d-dir outputs/masks_3d \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --output outputs/crack_groups_bbox.ply \
    --mode bbox
```

**Detailed mask meshes** (slow):
```bash
python -m src.export_group_meshes \
    --groups outputs/groups.json \
    --masks-3d-dir outputs/masks_3d \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --output outputs/crack_groups_detailed.ply \
    --mode detailed
```

Open in CloudCompare:
```bash
cloudcompare outputs/crack_groups_bbox.ply
```

---

## Complete Workflow Example

```bash
# Prerequisites (if not done)
python -m src.pipeline sfm --config configs/simple.yaml
python -m src.pipeline detect --config configs/simple.yaml

# Phase 3: Batch convert masks to 3D
python -m src.mask_to_3d \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --yolo-dir data/yolo_masks \
    --output-dir outputs/masks_3d

# Verify Phase 3
ls outputs/masks_3d/*.json | wc -l

# Phase 4: Group masks
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --depth-diff-threshold 0.2 \
    --min-confidence 0.25

# Verify Phase 4
python -c "import json; g=json.load(open('outputs/groups.json')); print(f'{len(g)} groups')"

# Phase 5: Visualize (Open3D - interactive)
python visualize_simple.py \
    --groups outputs/groups.json \
    --masks-3d-dir outputs/masks_3d \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --min-views 2 \
    --sample-rate 10
```

---

## Troubleshooting

### Issue 1: "Image not found in COLMAP model"

**Check COLMAP image names**:
```bash
python -c "
from src.colmap_io import read_images_binary
images = read_images_binary('data/sfm/sparse/0/images.bin')
print('COLMAP images:')
for img_id, img in list(images.items())[:5]:
    print(f'  {img.name}')
"
```

Make sure COLMAP image names match actual RGB filenames.

---

### Issue 2: "No valid depth values"

**Check depth image quality**:
```bash
python -c "
import cv2
import sys
depth = cv2.imread('data/depth/camera_DPT_<timestamp>.png', -1)
if depth is None:
    print('ERROR: Cannot read depth image')
    sys.exit(1)
print(f'Depth shape: {depth.shape}')
print(f'Valid pixels: {(depth > 0).sum() / depth.size * 100:.1f}%')
print(f'Depth range: {depth[depth>0].min()} - {depth[depth>0].max()}')
"
```

If valid pixels < 10%, depth image has issues.

---

### Issue 3: Empty groups or no results

**Verify YOLO output format**:
```bash
cat data/yolo_masks/camera_RGB_*.json | python -m json.tool | head -30
```

Should have structure:
```json
{
  "masks": [
    {
      "polygon": [[x1, y1], [x2, y2], ...],
      "score": 0.85
    }
  ]
}
```

---

### Issue 4: Groups with extent_3d = [0, 0, 0]

This happens when `num_masks = 1` (single detection, no spread).

**Filter by minimum views**:
```bash
# Only show groups seen in ≥2 views
python visualize_simple.py \
    --groups outputs/groups.json \
    ... \
    --min-views 2
```

Or filter programmatically:
```python
import json

groups = json.load(open('outputs/groups.json'))
filtered = [g for g in groups if g['num_views'] >= 2]
json.dump(filtered, open('outputs/groups_filtered.json', 'w'), indent=2)
```

---

## Understanding the Results

### Group Quality Metrics

- **num_views ≥ 3**: High confidence (seen from multiple angles)
- **num_views = 2**: Medium confidence
- **num_views = 1**: Low confidence (possible false positive)

- **confidence.mean > 0.7**: High YOLO confidence
- **confidence.mean 0.5-0.7**: Medium confidence
- **confidence.mean < 0.5**: Low confidence

### Spatial Interpretation

- **centroid_3d**: Global 3D position in SFM coordinate system (meters)
- **extent_3d**: Physical size [width, height, depth] in meters
- **depth_median**: Distance from camera (meters)

### Example Analysis

```python
import json

groups = json.load(open('outputs/groups.json'))

# Find largest cracks
large_cracks = sorted(groups, key=lambda g: g['extent_3d'][0] * g['extent_3d'][1], reverse=True)

print("Top 5 largest cracks:")
for i, crack in enumerate(large_cracks[:5]):
    area = crack['extent_3d'][0] * crack['extent_3d'][1]
    print(f"{i+1}. Position: {crack['centroid_3d']}")
    print(f"   Size: {crack['extent_3d']} m (area ≈ {area:.3f} m²)")
    print(f"   Confidence: {crack['confidence']['mean']:.3f}")
    print(f"   Views: {crack['num_views']}")
    print()
```

---

## Summary

**Pipeline**:
1. Phase 3: Mask → 3D (lightweight centroid-based)
2. Phase 4: Grouping (distance-based, no DBSCAN)
3. Phase 5: Visualization (SFM point cloud + colored groups)

**Advantages**:
- ✅ Fast (centroid-only, not full point clouds)
- ✅ Simple (no complex plane fitting or clustering)
- ✅ Reliable (density-independent grouping)
- ✅ Clear visualization (Open3D interactive viewer)

**Key Files**:
- `src/mask_to_3d.py`: Phase 3 implementation
- `src/group_masks_3d.py`: Phase 4 implementation
- `visualize_simple.py`: Phase 5 visualization (recommended)
