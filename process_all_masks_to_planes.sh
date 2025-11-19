#!/bin/bash

# Phase 3 일괄 처리: 모든 YOLO mask를 3D plane으로 변환
# 실제 프로젝트 파일명 패턴: camera_RGB_*.png, camera_DPT_*.png

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Phase 3: Mask → Plane 일괄 변환"
echo "========================================="

# 디렉토리 설정
RGB_DIR="data/rgb"
DEPTH_DIR="data/depth"
YOLO_DIR="data/yolo_masks"
OUTPUT_DIR="outputs/planes"

RGB_CALIB="calib/rgb_camera_info.json"
DEPTH_CALIB="calib/depth_camera_info.json"
EXTRINSIC="calib/extrinsic_depth_to_color.json"
COLMAP_MODEL="data/sfm/sparse/0"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 필수 파일 확인
echo -e "\n[체크] 필수 파일 확인 중..."

if [ ! -d "$RGB_DIR" ]; then
    echo -e "${RED}❌ RGB 디렉토리 없음: $RGB_DIR${NC}"
    exit 1
fi

if [ ! -d "$DEPTH_DIR" ]; then
    echo -e "${RED}❌ Depth 디렉토리 없음: $DEPTH_DIR${NC}"
    exit 1
fi

if [ ! -d "$YOLO_DIR" ]; then
    echo -e "${RED}❌ YOLO 마스크 디렉토리 없음: $YOLO_DIR${NC}"
    echo -e "${YELLOW}💡 먼저 YOLO 검출을 실행하세요:${NC}"
    echo "   python -m src.pipeline detect --config configs/simple.yaml"
    exit 1
fi

if [ ! -d "$COLMAP_MODEL" ]; then
    echo -e "${RED}❌ COLMAP 모델 없음: $COLMAP_MODEL${NC}"
    echo -e "${YELLOW}💡 먼저 SFM을 실행하세요:${NC}"
    echo "   python -m src.pipeline sfm --config configs/simple.yaml"
    exit 1
fi

echo -e "${GREEN}✅ 모든 필수 디렉토리 확인 완료${NC}"

# RGB 이미지 파일 찾기
RGB_FILES=("$RGB_DIR"/camera_RGB_*.png)

if [ ! -e "${RGB_FILES[0]}" ]; then
    echo -e "${RED}❌ RGB 이미지가 없습니다: $RGB_DIR/camera_RGB_*.png${NC}"
    exit 1
fi

TOTAL_FILES=${#RGB_FILES[@]}
echo -e "\n[발견] RGB 이미지: ${TOTAL_FILES}개"

# 처리 통계
PROCESSED=0
SKIPPED=0
FAILED=0

# 각 RGB 이미지에 대해 처리
for rgb_file in "${RGB_FILES[@]}"; do
    # 파일명에서 타임스탬프 추출
    # 예: camera_RGB_1758853283_533442048.png → 1758853283_533442048
    filename=$(basename "$rgb_file")
    timestamp=$(echo "$filename" | sed 's/camera_RGB_\(.*\)\.png/\1/')

    # 대응하는 파일들
    depth_file="$DEPTH_DIR/camera_DPT_${timestamp}.png"
    yolo_file="$YOLO_DIR/camera_RGB_${timestamp}.json"
    output_file="$OUTPUT_DIR/camera_RGB_${timestamp}.json"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "처리 중: camera_RGB_${timestamp}.png"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Depth 파일 확인
    if [ ! -f "$depth_file" ]; then
        echo -e "${YELLOW}⚠️  Depth 파일 없음: $(basename "$depth_file")${NC}"
        ((SKIPPED++))
        continue
    fi

    # YOLO 결과 확인
    if [ ! -f "$yolo_file" ]; then
        echo -e "${YELLOW}⚠️  YOLO 결과 없음: $(basename "$yolo_file")${NC}"
        ((SKIPPED++))
        continue
    fi

    # Plane 변환 실행
    echo -e "${GREEN}🔄 변환 시작...${NC}"

    if python -m src.mask_to_plane \
        --rgb-calib "$RGB_CALIB" \
        --depth-calib "$DEPTH_CALIB" \
        --extrinsic "$EXTRINSIC" \
        --colmap-model "$COLMAP_MODEL" \
        --image "$rgb_file" \
        --depth "$depth_file" \
        --yolo-results "$yolo_file" \
        --output "$output_file"; then

        echo -e "${GREEN}✅ 성공: $(basename "$output_file")${NC}"
        ((PROCESSED++))

        # 결과 요약 출력
        if [ -f "$output_file" ]; then
            python -c "
import json
try:
    with open('$output_file') as f:
        planes = json.load(f)
    print(f'   📊 평면 개수: {len(planes)}')
    if len(planes) > 0:
        avg_planarity = sum(p['quality']['planarity'] for p in planes) / len(planes)
        avg_coverage = sum(p['quality']['depth_coverage'] for p in planes) / len(planes)
        print(f'   📏 평균 planarity: {avg_planarity:.3f}')
        print(f'   📏 평균 coverage: {avg_coverage:.3f}')
except Exception as e:
    print(f'   ⚠️  결과 확인 실패: {e}')
"
        fi
    else
        echo -e "${RED}❌ 실패: 변환 중 오류 발생${NC}"
        ((FAILED++))
    fi
done

# 최종 통계
echo ""
echo "========================================="
echo "처리 완료!"
echo "========================================="
echo -e "총 파일:    ${TOTAL_FILES}"
echo -e "${GREEN}✅ 성공:     ${PROCESSED}${NC}"
echo -e "${YELLOW}⚠️  건너뜀:   ${SKIPPED}${NC}"
echo -e "${RED}❌ 실패:     ${FAILED}${NC}"
echo "========================================="
echo ""
echo "출력 위치: $OUTPUT_DIR/"
echo ""

if [ $PROCESSED -gt 0 ]; then
    echo -e "${GREEN}다음 단계:${NC}"
    echo "  python -m src.cluster_planes \\"
    echo "      --input $OUTPUT_DIR/*.json \\"
    echo "      --output outputs/clusters.json \\"
    echo "      --angle-threshold 15.0 \\"
    echo "      --distance-threshold 0.05 \\"
    echo "      --centroid-threshold 0.5"
    echo ""
fi

exit 0
