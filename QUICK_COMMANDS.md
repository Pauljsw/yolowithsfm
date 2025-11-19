# Quick Command Reference

Complete pipeline in 3 commands:

## Phase 3: Mask → 3D (Batch)

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

## Phase 4: Grouping

```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --depth-diff-threshold 0.2 \
    --min-confidence 0.25
```

## Phase 5: Visualization (Open3D)

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
    --sample-rate 10
```

---

## Verification Commands

### Check Phase 3 output
```bash
ls outputs/masks_3d/*.json | wc -l
```

### Check Phase 4 output
```bash
python -c "import json; g=json.load(open('outputs/groups.json')); print(f'Total groups: {len(g)}')"
```

### Inspect groups
```bash
python -c "
import json
groups = json.load(open('outputs/groups.json'))
for i, g in enumerate(groups[:5]):
    print(f'Group {i+1}: {g[\"num_masks\"]} masks, {g[\"num_views\"]} views, extent={g[\"extent_3d\"]} m')
"
```

---

## Parameter Tuning

### More aggressive grouping (fewer, larger groups)
```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 1.0 \
    --depth-diff-threshold 0.5 \
    --min-confidence 0.25
```

### More conservative grouping (more, smaller groups)
```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.3 \
    --depth-diff-threshold 0.1 \
    --min-confidence 0.25
```

### Ignore depth similarity
```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --min-confidence 0.25 \
    --no-depth-similarity
```

---

## Alternative Visualizations

### Matplotlib (save to image)
```bash
python -m src.visualize_groups_3d \
    --colmap-model data/sfm/sparse/0 \
    --groups outputs/groups.json \
    --output outputs/visualization_3d.png
```

### Export to PLY for CloudCompare (bbox mode)
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

### Open in CloudCompare
```bash
cloudcompare outputs/crack_groups_bbox.ply
```

---

For detailed explanations, see [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
