# 빠른 시작 가이드: 평면 기반 파이프라인

## 1줄 요약

YOLO mask → Depth로 3D 변환 → 평면 fitting → coplanarity로 클러스터링

---

## 필수 입력 데이터

```
✅ RGB 이미지 (data/rgb/*.jpg)
✅ Depth 이미지 (data/depth/*.png, 16-bit, mm 단위)
✅ YOLO 검출 결과 (outputs/yolo_detections/*.json)
✅ COLMAP 카메라 포즈 (outputs/sfm/sparse/0/*.bin)
✅ 카메라 캘리브레이션 (calib/*.json) ← 이미 있음
```

---

## Phase별 실행 (3단계)

### Phase 3: 단일 이미지 테스트

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

**출력**: `outputs/planes/IMG_001.json` (각 mask의 평면 파라미터)

### Phase 3: 모든 이미지 일괄 처리

```bash
# 스크립트 생성
cat > process_all.sh << 'EOF'
#!/bin/bash
mkdir -p outputs/planes
for rgb in data/rgb/*.jpg; do
    name=$(basename "$rgb" .jpg)
    python -m src.mask_to_plane \
        --rgb-calib calib/rgb_camera_info.json \
        --depth-calib calib/depth_camera_info.json \
        --extrinsic calib/extrinsic_depth_to_color.json \
        --colmap-model outputs/sfm/sparse/0 \
        --image "$rgb" \
        --depth "data/depth/${name}.png" \
        --yolo-results "outputs/yolo_detections/${name}.json" \
        --output "outputs/planes/${name}.json"
done
EOF

chmod +x process_all.sh
./process_all.sh
```

**출력**: `outputs/planes/*.json` (모든 이미지의 평면 파라미터)

### Phase 4: 클러스터링

```bash
python -m src.cluster_planes \
    --input outputs/planes/*.json \
    --output outputs/clusters.json \
    --angle-threshold 15.0 \
    --distance-threshold 0.05 \
    --centroid-threshold 0.5
```

**출력**: `outputs/clusters.json` (균열 클러스터 정보)

---

## 전체 파이프라인 한번에

```bash
python run_plane_pipeline.py \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --yolo-dir outputs/yolo_detections \
    --colmap-model outputs/sfm/sparse/0 \
    --output-dir outputs/plane_clustering \
    --visualize
```

---

## 주요 파라미터 튜닝

### 클러스터가 너무 많이 나뉨 (over-segmentation)
```bash
--angle-threshold 20.0      # 15 → 20 (더 관대)
--distance-threshold 0.10   # 0.05 → 0.10
--centroid-threshold 1.0    # 0.5 → 1.0
```

### 클러스터가 너무 적음 (under-segmentation)
```bash
--angle-threshold 10.0      # 15 → 10 (더 엄격)
--distance-threshold 0.03   # 0.05 → 0.03
--centroid-threshold 0.3    # 0.5 → 0.3
```

### 노이즈 많은 검출 필터링
```bash
--min-planarity 0.8         # 0.7 → 0.8 (더 엄격)
--min-coverage 0.5          # 0.3 → 0.5
```

---

## 결과 확인

```bash
# 클러스터 개수와 면적
python -c "
import json
clusters = json.load(open('outputs/clusters.json'))
print(f'Clusters: {len(clusters)}')
print(f'Total area: {sum(c[\"plane\"][\"area_3d\"] for c in clusters):.2f} m²')
for i, c in enumerate(clusters):
    print(f'  C{i+1}: {c[\"num_masks\"]}masks, {c[\"plane\"][\"area_3d\"]:.2f}m²')
"

# 시각화 (matplotlib 필요)
python run_plane_pipeline.py --visualize  # ... 기타 옵션들
# → outputs/plane_clustering/cluster_visualization.png
```

---

## 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| "Image not found in COLMAP" | 파일명 불일치 | `read_images_binary()로 이름 확인` |
| "No valid 3D points" | Depth 없음 | `check_alignment.py` 실행 |
| Planarity 낮음 (<0.5) | 평면 아님 | YOLO 마스크 확인 또는 필터링 |
| Coverage 낮음 (<0.3) | Depth 정렬 문제 | `align_depth_to_rgb` 실행 |

---

## 상세 문서

- **PHASE_BY_PHASE_GUIDE.md**: 각 Phase 상세 설명
- **PLANE_PIPELINE.md**: 전체 아키텍처 및 설계
- **DENSE_PIPELINE.md**: 기존 DBSCAN 방식 (참고용)

---

## 샘플 데이터 생성 (테스트용)

```bash
python create_sample_data.py
# → outputs/yolo_detections_sample/*.json
# → outputs/sfm_sample/sparse/0/*.bin
```

실제 RGB/Depth 이미지만 준비하면 테스트 가능
