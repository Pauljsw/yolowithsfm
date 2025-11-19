# Lightweight Mask-based 3D Grouping Pipeline

## 개요

**핵심 아이디어**: DBSCAN도, 평면 fitting도 필요 없음. 단순히 mask의 중심점을 3D로 변환하고 거리로 그룹핑.

### 장점
- ✅ **경량**: Centroid만 계산 (전체 포인트 클라우드 X)
- ✅ **빠름**: O(n²) 거리 계산만
- ✅ **간단**: Depth 사용하지만 복잡한 정렬 불필요
- ✅ **강건**: Density-independent

### Phase 구성

```
Phase 3: Mask → 3D
  - Input: YOLO mask + Depth + COLMAP pose
  - Process: Centroid 계산 + Median depth 사용
  - Output: {centroid_3d, depth_median, bbox_3d, ...}

Phase 4: Mask-level Grouping
  - Input: 모든 mask의 3D 데이터
  - Process: Centroid 거리 + Depth 유사도
  - Output: Groups of masks

Phase 5: Measurement
  - Input: Grouped masks
  - Output: 크기, 위치, 심각도 등
```

---

## 실행 방법

### 전제조건

기존 파이프라인의 Phase 0, 4 완료:

```bash
# Phase 0: SFM
python -m src.pipeline sfm --config configs/simple.yaml

# Phase 4: YOLO 검출
python -m src.pipeline detect --config configs/simple.yaml
```

**확인**:
```bash
ls data/sfm/sparse/0/images.bin     # ✅ 있어야 함
ls data/yolo_masks/ | head          # ✅ camera_RGB_*.json 있어야 함
ls data/depth/ | head               # ✅ camera_DPT_*.png 있어야 함
```

---

### Phase 3: Mask → 3D (단일 이미지 테스트)

```bash
# 이미지 하나로 테스트
python -m src.mask_to_3d \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --colmap-model data/sfm/sparse/0 \
    --image data/rgb/camera_RGB_<timestamp>.png \
    --depth data/depth/camera_DPT_<timestamp>.png \
    --yolo-results data/yolo_masks/camera_RGB_<timestamp>.json \
    --output outputs/masks_3d/camera_RGB_<timestamp>.json
```

**출력 예시** (`outputs/masks_3d/camera_RGB_*.json`):
```json
[
  {
    "image": "camera_RGB_1758853283_533442048.png",
    "mask_idx": 0,
    "confidence": 0.85,
    "centroid_2d": [1920, 1080],
    "centroid_3d": [1.2, 2.3, -2.4],
    "depth_median": 2.45,
    "depth_std": 0.12,
    "mask_area_2d": 15000,
    "bbox_2d": [1800, 1000, 2040, 1160],
    "bbox_3d_size": [0.15, 0.15, 0.24],
    "depth_coverage": 0.78
  }
]
```

**확인**:
```bash
cat outputs/masks_3d/camera_RGB_*.json | python -m json.tool | head -30
```

---

### Phase 3: 모든 이미지 일괄 처리 (Batch Mode)

**한 줄 명령으로 모든 이미지 처리**:

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

**특징**:
- ✅ COLMAP 모델 한 번만 로드 (매우 빠름)
- ✅ Progress bar로 진행상황 표시
- ✅ 자동 파일 매칭 (timestamp 기반)
- ✅ 통계 요약 출력

**확인**:
```bash
ls outputs/masks_3d/*.json | wc -l  # 처리된 파일 개수

# 전체 통계
python -c "
import json
from pathlib import Path

total = 0
for f in Path('outputs/masks_3d').glob('*.json'):
    with open(f) as fp:
        total += len(json.load(fp))
print(f'Total masks: {total}')
"
```

---

### Phase 4: Grouping

```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --depth-diff-threshold 0.2 \
    --min-confidence 0.25
```

**파라미터**:
- `--distance-threshold 0.5`: Centroid 간 최대 거리 (미터)
- `--depth-diff-threshold 0.2`: Depth 차이 최대값 (미터)
- `--min-confidence 0.25`: 최소 신뢰도

**출력** (`outputs/groups.json`):
```json
[
  {
    "num_masks": 12,
    "num_views": 5,
    "centroid_3d": [1.2, 2.3, -2.4],
    "depth_median": 2.45,
    "extent_3d": [0.3, 0.4, 0.1],
    "area_2d_total": 180000,
    "confidence": {
      "max": 0.92,
      "mean": 0.78
    },
    "images": ["camera_RGB_001.png", "camera_RGB_002.png", ...],
    "masks": [...]
  }
]
```

**확인**:
```bash
python -c "
import json
groups = json.load(open('outputs/groups.json'))
print(f'Total groups: {len(groups)}')
for i, g in enumerate(groups):
    print(f'  G{i+1}: {g[\"num_masks\"]}masks, extent={g[\"extent_3d\"]}m')
"
```

---

## 파라미터 튜닝

### 그룹이 너무 많이 나뉨 (과분할)

```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 1.0 \      # 0.5 → 1.0 (더 관대)
    --depth-diff-threshold 0.5 \    # 0.2 → 0.5
    --min-confidence 0.25
```

### 그룹이 너무 적음 (과병합)

```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.3 \      # 0.5 → 0.3 (더 엄격)
    --depth-diff-threshold 0.1 \    # 0.2 → 0.1
    --min-confidence 0.25
```

### Depth 유사도 무시

```bash
python -m src.group_masks_3d \
    --input outputs/masks_3d/*.json \
    --output outputs/groups.json \
    --distance-threshold 0.5 \
    --min-confidence 0.25 \
    --no-depth-similarity           # Depth 체크 비활성화
```

---

## 문제 해결

### 1. "Image not found in COLMAP model"

**확인**:
```bash
python -c "
from src.colmap_io import read_images_binary
images = read_images_binary('data/sfm/sparse/0/images.bin')
print('COLMAP images:')
for img_id, img in list(images.items())[:5]:
    print(f'  {img.name}')
"
```

COLMAP 이미지 이름과 실제 파일명이 일치하는지 확인

### 2. "No valid depth values"

**확인**:
```bash
python -c "
import cv2
depth = cv2.imread('data/depth/camera_DPT_<timestamp>.png', -1)
print(f'Depth shape: {depth.shape}')
print(f'Valid pixels: {(depth > 0).sum() / depth.size * 100:.1f}%')
"
```

Valid < 10%이면 depth 이미지 문제

### 3. Empty results (0 masks processed)

**원인**: YOLO 검출 결과가 없거나 잘못된 경로

**확인**:
```bash
# YOLO 결과 확인
ls data/yolo_masks/ | head
cat data/yolo_masks/camera_RGB_*.json | head -20

# Depth 파일 확인
ls data/depth/ | head
```

---

## 다음 단계 (Phase 5: Measurement)

Groups에서 측정:

```python
import json

with open('outputs/groups.json') as f:
    groups = json.load(f)

for i, group in enumerate(groups):
    print(f"Group {i+1}:")
    print(f"  위치: {group['centroid_3d']}")
    print(f"  크기: {group['extent_3d']} m")
    print(f"  심각도: {group['num_views']} views")
    print(f"  신뢰도: {group['confidence']['mean']:.3f}")
```

---

## 요약

**입력**:
- RGB 이미지 (data/rgb/)
- Depth 이미지 (data/depth/)
- YOLO 마스크 (data/yolo_masks/)
- COLMAP 포즈 (data/sfm/sparse/0/)

**출력**:
- Mask 3D 데이터 (outputs/masks_3d/*.json)
- Groups (outputs/groups.json)

**핵심**:
- Lightweight: Centroid만 계산
- Fast: 간단한 거리 기반 그룹핑
- No DBSCAN, No plane fitting
