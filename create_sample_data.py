"""
샘플 데이터 생성 스크립트

테스트용으로 가상의 YOLO 검출 결과와 COLMAP 카메라 포즈를 생성합니다.
실제 데이터가 준비되기 전에 파이프라인을 테스트할 수 있습니다.
"""

import json
import numpy as np
from pathlib import Path
import sys

# read_write_model import 추가
sys.path.append('src')
from read_write_model import write_images_binary, write_cameras_binary, Image, Camera


def create_sample_yolo_detection(image_name: str, output_path: Path):
    """
    샘플 YOLO 검출 결과 생성

    2-3개의 가상 균열 마스크 생성
    """
    detections = [
        {
            "class": 0,
            "confidence": 0.85,
            "bbox": [100, 200, 300, 400],
            "segmentation": [
                [120, 220, 280, 220, 280, 380, 120, 380]  # 사각형 마스크
            ]
        },
        {
            "class": 0,
            "confidence": 0.72,
            "bbox": [500, 300, 700, 500],
            "segmentation": [
                [520, 320, 680, 320, 680, 480, 520, 480]
            ]
        },
        {
            "class": 0,
            "confidence": 0.65,
            "bbox": [1000, 800, 1200, 1000],
            "segmentation": [
                [1020, 820, 1180, 820, 1180, 980, 1020, 980]
            ]
        }
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"✅ Created: {output_path}")


def create_sample_colmap_poses(num_images: int, output_dir: Path):
    """
    샘플 COLMAP 카메라 포즈 생성

    원형 경로를 따라 배치된 카메라들 생성
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 카메라 내부 파라미터 (RGB 카메라와 동일)
    cameras = {}
    cameras[1] = Camera(
        id=1,
        model='OPENCV',
        width=3840,
        height=2160,
        params=np.array([2246.03, 2244.84, 1903.48, 1091.56,  # fx, fy, cx, cy
                        0.0779024, -0.106185, -0.000293252, -4.25309e-05])  # distortion
    )

    # 카메라 포즈들 (원형 경로)
    images = {}
    radius = 3.0  # 3m 반경

    for i in range(num_images):
        angle = 2 * np.pi * i / num_images

        # 카메라 위치 (원형)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.5  # 1.5m 높이

        # 중심을 향하는 방향
        look_at = np.array([0.0, 0.0, 1.5])
        camera_pos = np.array([x, y, z])
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)

        # 카메라 좌표계 구성
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Rotation matrix (world to camera)
        R = np.stack([right, -up, forward], axis=0)

        # Translation (world to camera)
        t = -R @ camera_pos

        # Quaternion 변환
        qvec = rotmat2qvec(R)

        images[i+1] = Image(
            id=i+1,
            qvec=qvec,
            tvec=t,
            camera_id=1,
            name=f"IMG_{i+1:03d}.jpg",
            xys=np.zeros((0, 2)),  # 빈 특징점
            point3D_ids=np.zeros(0, dtype=int)
        )

    # 파일로 저장
    write_cameras_binary(cameras, str(output_dir / 'cameras.bin'))
    write_images_binary(images, str(output_dir / 'images.bin'))

    # points3D.bin은 빈 파일 생성 (필요하지만 사용 안함)
    import struct
    with open(output_dir / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('Q', 0))  # 0개의 3D 포인트

    print(f"✅ Created COLMAP model in: {output_dir}")
    print(f"   - {num_images} camera poses")
    print(f"   - Circular trajectory (radius={radius}m)")


def rotmat2qvec(R):
    """Rotation matrix to quaternion (COLMAP convention)"""
    # Shepperd's method
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]]
    ]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]

    if qvec[0] < 0:
        qvec = -qvec

    return qvec


def main():
    print("="*60)
    print("샘플 데이터 생성")
    print("="*60)

    num_images = 10

    # 1. YOLO 검출 결과 생성
    print(f"\n[1/2] YOLO 검출 결과 생성 ({num_images}개 이미지)...")
    yolo_dir = Path('outputs/yolo_detections_sample')

    for i in range(1, num_images + 1):
        image_name = f"IMG_{i:03d}.jpg"
        output_path = yolo_dir / f"IMG_{i:03d}.json"
        create_sample_yolo_detection(image_name, output_path)

    # 2. COLMAP 카메라 포즈 생성
    print(f"\n[2/2] COLMAP 카메라 포즈 생성 ({num_images}개 포즈)...")
    colmap_dir = Path('outputs/sfm_sample/sparse/0')
    create_sample_colmap_poses(num_images, colmap_dir)

    # 완료 메시지
    print("\n" + "="*60)
    print("✅ 샘플 데이터 생성 완료!")
    print("="*60)
    print("\n다음 단계:")
    print("1. RGB 이미지 준비: data/rgb/IMG_001.jpg ~ IMG_010.jpg")
    print("2. Depth 이미지 준비: data/depth/IMG_001.png ~ IMG_010.png")
    print("3. Phase 3 실행 예시:")
    print()
    print("   python -m src.mask_to_plane \\")
    print("       --rgb-calib calib/rgb_camera_info.json \\")
    print("       --depth-calib calib/depth_camera_info.json \\")
    print("       --extrinsic calib/extrinsic_depth_to_color.json \\")
    print("       --colmap-model outputs/sfm_sample/sparse/0 \\")
    print("       --image data/rgb/IMG_001.jpg \\")
    print("       --depth data/depth/IMG_001.png \\")
    print("       --yolo-results outputs/yolo_detections_sample/IMG_001.json \\")
    print("       --output outputs/planes_sample/IMG_001.json")
    print()
    print("="*60)


if __name__ == '__main__':
    main()
