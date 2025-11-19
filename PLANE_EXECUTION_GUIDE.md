# 평면 기반 파이프라인 실행 가이드

## 📁 실제 프로젝트 구조

```
yolowithsfm/
├── data/
│   ├── rgb/                         # RGB 이미지
│   │   ├── camera_RGB_*.png        # 예: camera_RGB_1758853283_533442048.png
│   │   └── ...
│   ├── depth/                       # Depth 이미지 (RGB와 타임스탬프 매칭)
│   │   ├── camera_DPT_*.png        # 예: camera_DPT_1758853283_533442048.png
│   │   └── ...
│   ├── yolo_masks/                  # YOLO 검출 결과 (Phase 4에서 생성)
│   │   ├── camera_RGB_*.json
│   │   └── ...
│   └── sfm/                         # COLMAP SFM 결과 (Phase 0에서 생성)
│       └── sparse/0/
│           ├── images.bin
│           ├── cameras.bin
│           └── points3D.bin
├── calib/                           # 카메라 캘리브레이션 (이미 있음)
│   ├── rgb_camera_info.json
│   ├── depth_camera_info.json
│   └── extrinsic_depth_to_color.json
└── outputs/                         # 출력 디렉토리
    └── plane_clustering/            # 평면 클러스터링 결과
        ├── planes/                  # 이미지별 평면 파라미터
        ├── clusters.json            # 최종 클러스터
        └── summary.txt              # 텍스트 리포트
```

---

## 🚀 Phase별 실행 (평면 기반)

### 전제 조건 확인

기존 파이프라인의 **Phase 0**과 **Phase 4**가 완료되어야 합니다:

```bash
# Phase 0: SFM (COLMAP) - 카메라 포즈 추정
python -m src.pipeline sfm --config configs/simple.yaml

# Phase 4: YOLO Inference - 균열 검출
python -m src.pipeline detect --config configs/simple.yaml
```

**확인 방법**:
```bash
# SFM 결과 확인
ls -lh data/sfm/sparse/0/
# ✅ images.bin, cameras.bin, points3D.bin 있어야 함

# YOLO 결과 확인
ls data/yolo_masks/ | head -5
# ✅ camera_RGB_*.json 파일들이 있어야 함
```

---

## Phase 3 (NEW): Mask → Plane 변환

### 방법 1: 단일 이미지 테스트 (권장)

먼저 이미지 1개로 테스트해보세요:

```bash
# RGB 파일 하나 선택 (예시)
RGB_FILE=$(ls data/rgb/camera_RGB_*.png | head -1)
BASENAME=$(basename "$RGB_FILE" .png)
TIMESTAMP=$(echo "$BASENAME" | sed 's/camera_RGB_//')

# 실행
python -m src.mask_to_plane \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --extrinsic calib/extrinsic_depth_to_color.json \
    --colmap-model data/sfm/sparse/0 \
    --image "data/rgb/camera_RGB_${TIMESTAMP}.png" \
    --depth "data/depth/camera_DPT_${TIMESTAMP}.png" \
    --yolo-results "data/yolo_masks/camera_RGB_${TIMESTAMP}.json" \
    --output "outputs/planes/camera_RGB_${TIMESTAMP}.json"
```

**출력 확인**:
```bash
# JSON 파일 확인
cat outputs/planes/camera_RGB_${TIMESTAMP}.json | python -m json.tool | head -40

# 결과 요약
python -c "
import json
with open('outputs/planes/camera_RGB_${TIMESTAMP}.json') as f:
    planes = json.load(f)
print(f'✅ 평면 개수: {len(planes)}')
for i, p in enumerate(planes):
    print(f'  Mask {i}: planarity={p[\"quality\"][\"planarity\"]:.3f}, '
          f'coverage={p[\"quality\"][\"depth_coverage\"]:.3f}')
"
```

**해석**:
- `planarity ≥ 0.7`: 평면에 잘 맞음 (좋음)
- `planarity < 0.5`: 평면 아닐 가능성 (확인 필요)
- `coverage ≥ 0.3`: Depth 커버리지 충분 (좋음)
- `coverage < 0.2`: Depth 부족 (정렬 문제 가능)

### 방법 2: 모든 이미지 일괄 처리

테스트가 성공하면 전체 이미지를 처리하세요:

```bash
# 배치 스크립트 실행
./process_all_masks_to_planes.sh
```

**출력 예시**:
```
=========================================
Phase 3: Mask → Plane 일괄 변환
=========================================

[발견] RGB 이미지: 45개

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
처리 중: camera_RGB_1758853283_533442048.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 변환 시작...
✅ 성공: camera_RGB_1758853283_533442048.json
   📊 평면 개수: 3
   📏 평균 planarity: 0.873
   📏 평균 coverage: 0.682

...

=========================================
처리 완료!
=========================================
총 파일:    45
✅ 성공:     42
⚠️  건너뜀:   2
❌ 실패:     1
=========================================
```

**결과 확인**:
```bash
# 생성된 파일 개수
ls outputs/planes/*.json | wc -l

# 전체 통계
python -c "
import json
from pathlib import Path

total_planes = 0
high_quality = 0

for plane_file in Path('outputs/planes').glob('*.json'):
    with open(plane_file) as f:
        planes = json.load(f)
    total_planes += len(planes)
    high_quality += sum(1 for p in planes
                       if p['quality']['planarity'] >= 0.7
                       and p['quality']['depth_coverage'] >= 0.3)

print(f'총 평면: {total_planes}')
print(f'고품질: {high_quality} ({high_quality/total_planes*100:.1f}%)')
"
```

---

## Phase 4 (NEW): Plane 클러스터링

모든 평면을 coplanarity 기준으로 그룹핑합니다:

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
- `--angle-threshold 15.0`: 법선 각도 최대 15° (평행 판단)
- `--distance-threshold 0.05`: 평면 간 거리 최대 5cm
- `--centroid-threshold 0.5`: 중심점 거리 최대 50cm
- `--min-planarity 0.7`: Planarity 0.7 미만 필터링
- `--min-coverage 0.3`: Coverage 0.3 미만 필터링

**출력 확인**:
```bash
# 클러스터 요약
python -c "
import json
with open('outputs/clusters.json') as f:
    clusters = json.load(f)

print('='*60)
print(f'총 클러스터: {len(clusters)}')
print(f'총 면적: {sum(c[\"plane\"][\"area_3d\"] for c in clusters):.2f} m²')
print('='*60)

for i, c in enumerate(clusters):
    print(f'\nCluster {i+1}:')
    print(f'  Masks: {c[\"num_masks\"]} (from {c[\"num_views\"]} views)')
    print(f'  Area: {c[\"plane\"][\"area_3d\"]:.2f} m²')
    print(f'  Confidence: {c[\"confidence\"][\"mean\"]:.3f}')
    print(f'  Normal: {c[\"plane\"][\"normal\"]}')
"
```

---

## 전체 파이프라인 한번에 (자동화)

Phase 3 + Phase 4를 한번에:

```bash
python run_plane_pipeline.py \
    --visualize
```

**기본값** (프로젝트 구조에 맞춰져 있음):
- `--rgb-dir data/rgb`
- `--depth-dir data/depth`
- `--yolo-dir data/yolo_masks`
- `--colmap-model data/sfm/sparse/0`
- `--output-dir outputs/plane_clustering`

**출력**:
```
outputs/plane_clustering/
├── planes/
│   ├── camera_RGB_*.json      # 이미지별 평면 파라미터
│   └── ...
├── clusters.json              # 클러스터 데이터
├── summary.txt                # 텍스트 리포트
└── cluster_visualization.png  # 3D 시각화 (--visualize 사용 시)
```

---

## 파라미터 튜닝

### 클러스터가 너무 많이 나뉨 (과분할)

```bash
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 20.0 \       # 15 → 20 (더 관대)
    --distance-threshold 0.10 \    # 0.05 → 0.10
    --centroid-threshold 1.0       # 0.5 → 1.0
```

### 클러스터가 너무 적음 (과병합)

```bash
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 10.0 \       # 15 → 10 (더 엄격)
    --distance-threshold 0.03 \    # 0.05 → 0.03
    --centroid-threshold 0.3       # 0.5 → 0.3
```

### 노이즈 많은 검출 필터링

```bash
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --min-planarity 0.8 \          # 0.7 → 0.8 (더 엄격)
    --min-coverage 0.5             # 0.3 → 0.5
```

---

## 문제 해결

### 1. "Image not found in COLMAP model"

**원인**: COLMAP의 이미지 이름과 실제 파일명 불일치

**확인**:
```bash
python -c "
from src.read_write_model import read_images_binary
images = read_images_binary('data/sfm/sparse/0/images.bin')
print('COLMAP에 등록된 이미지:')
for img_id, img in list(images.items())[:5]:
    print(f'  {img.name}')
"

ls data/rgb/ | head -5
```

### 2. "No valid 3D points (no depth coverage)"

**원인**: Depth 이미지에 유효한 값이 없거나 정렬 문제

**확인**:
```bash
python -c "
import cv2
depth = cv2.imread('data/depth/camera_DPT_<timestamp>.png', -1)
print(f'Shape: {depth.shape}')
print(f'Min: {depth.min()}, Max: {depth.max()}')
print(f'Valid ratio: {(depth > 0).sum() / depth.size * 100:.1f}%')
"
```

**해결**: Valid ratio < 10%이면 depth 정렬 문제
```bash
python check_alignment.py  # 정렬 상태 확인
```

### 3. Planarity 낮음 (<0.5)

**원인**: 균열이 실제로 평면이 아니거나 YOLO 마스크 불량

**해결**:
- YOLO 검출 결과 시각적 확인
- `--min-planarity` threshold 조정
- 곡면 크랙인 경우 제외 (현재 평면 가정)

### 4. Coverage 낮음 (<0.3)

**원인**: Depth 이미지와 RGB 정렬 문제

**해결**:
```bash
# Depth 정렬 상태 확인
python test_depth_format.py

# 필요시 정렬 (현재는 pre-aligned로 가정)
```

---

## 다음 단계

클러스터링 결과 활용:

1. **리포트 생성**: `outputs/clusters.json` → PDF/HTML 리포트
2. **시각화**: 3D 뷰어에서 균열 위치 확인
3. **측정**: 클러스터별 면적, 길이 계산
4. **우선순위**: Confidence, area, view 수로 심각도 판단
5. **BIM 통합**: 평면 파라미터를 건물 모델에 매핑

---

## 참고 문서

- **PLANE_PIPELINE.md**: 전체 아키텍처 및 설계 철학
- **QUICK_START.md**: 빠른 참조 (명령어만)
- **DENSE_PIPELINE.md**: 기존 DBSCAN 방식 (비교용)
