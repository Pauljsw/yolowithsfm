#!/bin/bash

# Phase 3: Mask → 3D 일괄 처리 (Lightweight)
# Centroid + median depth만 계산

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "Phase 3: Mask → 3D 변환 (Lightweight)"
echo "========================================="

# 디렉토리 설정
RGB_DIR="data/rgb"
DEPTH_DIR="data/depth"
YOLO_DIR="data/yolo_masks"
OUTPUT_DIR="outputs/masks_3d"

RGB_CALIB="calib/rgb_camera_info.json"
DEPTH_CALIB="calib/depth_camera_info.json"
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
    echo -e "${YELLOW}💡 Depth 이미지가 필요합니다${NC}"
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

    # 3D 변환 실행
    echo -e "${GREEN}🔄 변환 시작...${NC}"

    if python -m src.mask_to_3d \
        --rgb-calib "$RGB_CALIB" \
        --depth-calib "$DEPTH_CALIB" \
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
        masks = json.load(f)
    print(f'   📊 Masks: {len(masks)}개')
    if len(masks) > 0:
        avg_depth = sum(m['depth_median'] for m in masks) / len(masks)
        avg_coverage = sum(m['depth_coverage'] for m in masks) / len(masks)
        print(f'   📏 평균 depth: {avg_depth:.2f}m')
        print(f'   📏 평균 coverage: {avg_coverage:.1%}')
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
    echo -e "${GREEN}다음 단계 (Phase 4 - Grouping):${NC}"
    echo "  python -m src.group_masks_3d \\"
    echo "      --input $OUTPUT_DIR/*.json \\"
    echo "      --output outputs/groups.json \\"
    echo "      --distance-threshold 0.5 \\"
    echo "      --depth-diff-threshold 0.2 \\"
    echo "      --min-confidence 0.25"
    echo ""
fi

exit 0
