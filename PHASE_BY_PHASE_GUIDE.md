# Phase별 실행 가이드: 평면 기반 파이프라인

## 전제 조건

아래 데이터가 준비되어 있어야 합니다:

```
yolowithsfm/
├── data/
│   ├── rgb/               # RGB 이미지들
│   │   ├── IMG_001.jpg
│   │   ├── IMG_002.jpg
│   │   └── ...
│   └── depth/             # Depth 이미지들 (RGB와 파일명 일치)
│       ├── IMG_001.png
│       ├── IMG_002.png
│       └── ...
├── calib/                 # 카메라 캘리브레이션 (이미 있음)
│   ├── rgb_camera_info.json
│   ├── depth_camera_info.json
│   └── extrinsic_depth_to_color.json
├── outputs/
│   ├── yolo_detections/   # YOLO 결과 (Phase 1에서 생성)
│   │   ├── IMG_001.json
│   │   ├── IMG_002.json
│   │   └── ...
│   └── sfm/               # COLMAP 결과 (Phase 2에서 생성)
│       └── sparse/0/
│           ├── images.bin
│           ├── cameras.bin
│           └── points3D.bin
```

---

## Phase 0: YOLO 검출 (기존 그대로)

**목적**: RGB 이미지에서 균열 마스크 검출

**실행**:
```bash
# YOLOv11 실행 (기존 방식)
yolo task=segment mode=predict \
    model=yolov11n-seg.pt \
    source=data/rgb \
    save_txt=True \
    save_conf=True \
    project=outputs \
    name=yolo_detections
```

**출력 형식** (`outputs/yolo_detections/IMG_001.json`):
```json
[
  {
    "class": 0,
    "confidence": 0.85,
    "bbox": [100, 200, 300, 400],
    "segmentation": [[x1, y1, x2, y2, ..., xn, yn]]
  }
]
```

**확인 방법**:
```bash
ls -lh outputs/yolo_detections/
# IMG_001.json, IMG_002.json 등이 있어야 함

cat outputs/yolo_detections/IMG_001.json | head -20
# JSON 형식 확인
```

---

## Phase 1: COLMAP SFM (기존 그대로)

**목적**: RGB 이미지들로부터 카메라 포즈 추정 (전역 좌표계 생성)

**실행**:
```bash
# Feature extraction
colmap feature_extractor \
    --database_path outputs/sfm/database.db \
    --image_path data/rgb

# Feature matching
colmap exhaustive_matcher \
    --database_path outputs/sfm/database.db

# Sparse reconstruction
colmap mapper \
    --database_path outputs/sfm/database.db \
    --image_path data/rgb \
    --output_path outputs/sfm/sparse
```

**출력**:
- `outputs/sfm/sparse/0/images.bin`: 각 이미지의 카메라 포즈 (R, t)
- `outputs/sfm/sparse/0/cameras.bin`: 카메라 내부 파라미터
- `outputs/sfm/sparse/0/points3D.bin`: 3D 특징점 (이번 파이프라인에서는 사용 안함)

**확인 방법**:
```bash
ls -lh outputs/sfm/sparse/0/
# images.bin, cameras.bin, points3D.bin 있어야 함

# 바이너리를 텍스트로 변환해서 확인
python -c "
from src.read_write_model import read_images_binary
images = read_images_binary('outputs/sfm/sparse/0/images.bin')
print(f'Total images: {len(images)}')
for img_id, img in list(images.items())[:3]:
    print(f'  {img.name}: qvec={img.qvec[:2]}... tvec={img.tvec[:2]}...')
"
```

---

## Phase 2: Depth 정렬 확인 (선택사항)

**목적**: Depth 이미지가 RGB와 정렬되어 있는지 확인

**실행**:
```bash
# 정렬 상태 테스트
python test_depth_format.py

# 시각적 확인
python check_alignment.py
```

**결과 해석**:
- Coverage > 70% → Depth가 이미 정렬됨 (그대로 사용)
- Coverage < 30% → Depth가 raw 형식 (정렬 필요)

**정렬이 필요한 경우**:
```bash
# 모든 depth 이미지 정렬
python -m src.align_depth_to_rgb \
    --rgb-dir data/rgb \
    --depth-dir data/depth_raw \
    --output-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --extrinsic calib/extrinsic_depth_to_color.json
```

---

## **Phase 3 (NEW): Mask → Plane 변환**

### 3-1. 단일 이미지 테스트

**목적**: YOLO mask를 3D 평면으로 변환 (depth + COLMAP pose 사용)

**실행**:
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

**입력**:
- `data/rgb/IMG_001.jpg`: RGB 이미지
- `data/depth/IMG_001.png`: 정렬된 depth 이미지 (16-bit, mm 단위)
- `outputs/yolo_detections/IMG_001.json`: YOLO 검출 결과
- `outputs/sfm/sparse/0/images.bin`: COLMAP 카메라 포즈

**출력** (`outputs/planes/IMG_001.json`):
```json
[
  {
    "image": "IMG_001.jpg",
    "mask_idx": 0,
    "confidence": 0.85,
    "plane": {
      "normal": [0.02, -0.99, 0.12],
      "d": -2.45,
      "center": [1.2, 2.3, -2.4],
      "area_3d": 0.15,
      "num_points": 4823
    },
    "quality": {
      "planarity": 0.95,
      "depth_coverage": 0.78
    }
  },
  {
    "image": "IMG_001.jpg",
    "mask_idx": 1,
    "confidence": 0.72,
    ...
  }
]
```

**확인 방법**:
```bash
cat outputs/planes/IMG_001.json | python -m json.tool | head -30

# planarity와 depth_coverage 확인
python -c "
import json
with open('outputs/planes/IMG_001.json') as f:
    planes = json.load(f)
print(f'Total masks in IMG_001: {len(planes)}')
for i, p in enumerate(planes):
    q = p['quality']
    print(f'  Mask {i}: planarity={q[\"planarity\"]:.3f}, coverage={q[\"depth_coverage\"]:.3f}')
"
```

**파라미터 설명**:
- `plane.normal`: 평면의 법선 벡터 (단위 벡터)
- `plane.d`: 평면 방정식 ax + by + cz + d = 0의 d
- `plane.center`: 3D 포인트들의 중심점 (전역 좌표)
- `plane.area_3d`: 추정된 3D 표면 적 (m²)
- `quality.planarity`: 점들이 평면에 얼마나 잘 맞는지 (0~1, 높을수록 좋음)
- `quality.depth_coverage`: 마스크 영역 중 유효한 depth 비율 (0~1)

### 3-2. 모든 이미지 일괄 처리

**스크립트 작성** (`process_all_planes.sh`):
```bash
#!/bin/bash

mkdir -p outputs/planes

for rgb_file in data/rgb/*.jpg; do
    filename=$(basename "$rgb_file" .jpg)
    depth_file="data/depth/${filename}.png"
    yolo_file="outputs/yolo_detections/${filename}.json"
    output_file="outputs/planes/${filename}.json"

    if [ ! -f "$depth_file" ]; then
        echo "❌ Missing depth: $depth_file"
        continue
    fi

    if [ ! -f "$yolo_file" ]; then
        echo "❌ Missing YOLO: $yolo_file"
        continue
    fi

    echo "🔄 Processing: $filename"

    python -m src.mask_to_plane \
        --rgb-calib calib/rgb_camera_info.json \
        --depth-calib calib/depth_camera_info.json \
        --extrinsic calib/extrinsic_depth_to_color.json \
        --colmap-model outputs/sfm/sparse/0 \
        --image "$rgb_file" \
        --depth "$depth_file" \
        --yolo-results "$yolo_file" \
        --output "$output_file"

    echo "✅ Done: $output_file"
done

echo ""
echo "==================================="
echo "Total plane files: $(ls outputs/planes/*.json | wc -l)"
echo "==================================="
```

**실행**:
```bash
chmod +x process_all_planes.sh
./process_all_planes.sh
```

**확인**:
```bash
ls -lh outputs/planes/
# IMG_001.json, IMG_002.json ... 등이 생성됨

# 전체 통계
python -c "
import json
from pathlib import Path

total_masks = 0
high_quality = 0

for plane_file in Path('outputs/planes').glob('*.json'):
    with open(plane_file) as f:
        planes = json.load(f)
    total_masks += len(planes)
    high_quality += sum(1 for p in planes
                       if p['quality']['planarity'] >= 0.7
                       and p['quality']['depth_coverage'] >= 0.3)

print(f'Total masks: {total_masks}')
print(f'High quality: {high_quality} ({high_quality/total_masks*100:.1f}%)')
"
```

---

## **Phase 4 (NEW): Plane 클러스터링**

**목적**: 모든 평면들을 coplanarity 기준으로 그룹핑 (같은 표면의 균열 병합)

**실행**:
```bash
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 15.0 \
    --distance-threshold 0.05 \
    --centroid-threshold 0.5 \
    --min-planarity 0.7 \
    --min-coverage 0.3
```

**파라미터 설명**:
- `--angle-threshold 15.0`: 법선 벡터 간 최대 각도 (도, degrees)
  - 두 평면의 법선이 15° 이내면 평행하다고 판단
  - 값이 클수록 더 많이 병합 (과도하게 병합 위험)
  - 값이 작을수록 엄격 (과도하게 분리 위험)

- `--distance-threshold 0.05`: 평면 간 수직 거리 (미터)
  - 평행한 두 평면이 5cm 이내면 같은 평면으로 판단
  - 벽 두께, 측정 오차 고려

- `--centroid-threshold 0.5`: 중심점 간 거리 (미터)
  - 평면이 평행하고 가까워도 중심이 50cm 이상 떨어지면 별개
  - 같은 벽의 다른 위치 균열 구분

- `--min-planarity 0.7`: 최소 평면 적합도
  - 0.7 미만은 필터링 (점들이 평면에 잘 안 맞음)

- `--min-coverage 0.3`: 최소 depth 커버리지
  - 마스크의 30% 미만만 depth 있으면 필터링 (신뢰도 낮음)

**출력** (`outputs/clusters.json`):
```json
[
  {
    "num_masks": 12,
    "num_views": 5,
    "plane": {
      "normal": [0.02, -0.99, 0.12],
      "d": -2.45,
      "center": [1.2, 2.3, -2.4],
      "area_3d": 0.85
    },
    "confidence": {
      "max": 0.92,
      "mean": 0.78
    },
    "images": ["IMG_001.jpg", "IMG_002.jpg", "IMG_005.jpg", ...],
    "masks": [
      {"image": "IMG_001.jpg", "mask_idx": 0, "confidence": 0.85},
      {"image": "IMG_001.jpg", "mask_idx": 2, "confidence": 0.72},
      ...
    ]
  },
  {
    "num_masks": 8,
    "num_views": 3,
    ...
  }
]
```

**확인**:
```bash
cat outputs/clusters.json | python -m json.tool | head -50

# 클러스터 요약
python -c "
import json

with open('outputs/clusters.json') as f:
    clusters = json.load(f)

print('='*60)
print(f'Total clusters: {len(clusters)}')
print(f'Total area: {sum(c[\"plane\"][\"area_3d\"] for c in clusters):.2f} m²')
print('='*60)

for i, c in enumerate(clusters):
    print(f'\nCluster {i+1}:')
    print(f'  Masks: {c[\"num_masks\"]} from {c[\"num_views\"]} views')
    print(f'  Area: {c[\"plane\"][\"area_3d\"]:.2f} m²')
    print(f'  Confidence: {c[\"confidence\"][\"mean\"]:.3f}')
"
```

---

## Phase 5: 결과 시각화 및 분석

### 5-1. 텍스트 리포트 생성

**실행**:
```bash
python -c "
import json

with open('outputs/clusters.json') as f:
    clusters = json.load(f)

with open('outputs/summary.txt', 'w') as f:
    f.write('='*80 + '\n')
    f.write('CRACK CLUSTERING SUMMARY\n')
    f.write('='*80 + '\n\n')

    f.write(f'Total clusters: {len(clusters)}\n')
    f.write(f'Total area: {sum(c[\"plane\"][\"area_3d\"] for c in clusters):.2f} m²\n\n')

    for i, c in enumerate(clusters):
        f.write(f'\nCluster {i+1}:\n')
        f.write(f'  Masks: {c[\"num_masks\"]}, Views: {c[\"num_views\"]}\n')
        f.write(f'  Area: {c[\"plane\"][\"area_3d\"]:.2f} m²\n')
        f.write(f'  Normal: {c[\"plane\"][\"normal\"]}\n')
        f.write(f'  Images: {c[\"images\"]}\n')

print('✅ Saved to outputs/summary.txt')
"

cat outputs/summary.txt
```

### 5-2. 3D 시각화

**실행**:
```bash
python -c "
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('outputs/clusters.json') as f:
    clusters = json.load(f)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))

for i, cluster in enumerate(clusters):
    center = cluster['plane']['center']
    normal = cluster['plane']['normal']
    area = cluster['plane']['area_3d']

    ax.scatter(center[0], center[1], center[2],
               color=colors[i], s=area*1000, alpha=0.6,
               label=f'C{i+1} ({cluster[\"num_masks\"]}m)')

    ax.quiver(center[0], center[1], center[2],
              normal[0], normal[1], normal[2],
              length=0.2, color=colors[i], alpha=0.8)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Crack Clusters (3D)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('outputs/clusters_3d.png', dpi=150, bbox_inches='tight')
print('✅ Saved to outputs/clusters_3d.png')
"
```

---

## 전체 파이프라인 한번에 실행 (자동화)

위 모든 단계를 한번에:

```bash
python run_plane_pipeline.py \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --yolo-dir outputs/yolo_detections \
    --colmap-model outputs/sfm/sparse/0 \
    --output-dir outputs/plane_clustering \
    --angle-threshold 15.0 \
    --distance-threshold 0.05 \
    --centroid-threshold 0.5 \
    --min-planarity 0.7 \
    --min-coverage 0.3 \
    --visualize
```

---

## 트러블슈팅

### 문제: "Image not found in COLMAP model"

**원인**: COLMAP의 이미지 이름과 실제 파일 이름 불일치

**해결**:
```bash
# COLMAP에 등록된 이미지 이름 확인
python -c "
from src.read_write_model import read_images_binary
images = read_images_binary('outputs/sfm/sparse/0/images.bin')
for img_id, img in images.items():
    print(img.name)
" > colmap_images.txt

# 실제 파일 이름 확인
ls data/rgb/ > actual_images.txt

# 차이 확인
diff colmap_images.txt actual_images.txt
```

### 문제: "No valid 3D points (no depth coverage)"

**원인**: Depth 이미지에 유효한 값이 없음

**해결**:
```bash
# Depth 이미지 확인
python -c "
import cv2
depth = cv2.imread('data/depth/IMG_001.png', -1)
print(f'Shape: {depth.shape}')
print(f'Min: {depth.min()}, Max: {depth.max()}')
print(f'Valid ratio: {(depth > 0).sum() / depth.size * 100:.1f}%')
"

# Valid ratio < 10% → Depth 정렬 문제 → check_alignment.py 재실행
```

### 문제: "Too many clusters" (과도하게 분리됨)

**원인**: Threshold가 너무 엄격

**해결**:
```bash
# Threshold 완화
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 20.0 \        # 15 → 20
    --distance-threshold 0.10 \     # 0.05 → 0.10
    --centroid-threshold 1.0        # 0.5 → 1.0
```

### 문제: "Too few clusters" (과도하게 병합됨)

**원인**: Threshold가 너무 관대

**해결**:
```bash
# Threshold 강화
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 10.0 \        # 15 → 10
    --distance-threshold 0.03 \     # 0.05 → 0.03
    --centroid-threshold 0.3        # 0.5 → 0.3
```

---

## 다음 단계

클러스터링 결과를 활용하여:
1. **보고서 생성**: 각 클러스터별 균열 위치, 크기, 심각도
2. **CAD 통합**: 평면 파라미터를 BIM 모델에 매핑
3. **시간 추적**: 동일 클러스터를 시간에 따라 추적 (균열 성장 모니터링)
4. **우선순위 지정**: 면적, 뷰 수, confidence 기반 심각도 순위
