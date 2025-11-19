"""
Test if depth PNG is already aligned or raw
"""
import cv2
import numpy as np
import json

def test_depth_format():
    print("=" * 80)
    print("Depth Format Detection Test")
    print("=" * 80)

    # Load images
    rgb = cv2.imread('data/rgb/sample.png')
    depth = cv2.imread('data/depth/sample.png', -1)

    # Load calibration
    with open('calib/depth_camera_info.json') as f:
        depth_calib = json.load(f)
    with open('calib/rgb_camera_info.json') as f:
        rgb_calib = json.load(f)

    print("\n1. Resolution comparison:")
    print(f"   RGB: {rgb.shape[:2]}")
    print(f"   Depth: {depth.shape}")

    # Test 1: Aspect ratio
    print("\n2. Aspect ratio test:")
    rgb_aspect = rgb.shape[1] / rgb.shape[0]  # 3840/2160 = 1.778
    depth_aspect = depth.shape[1] / depth.shape[0]  # 512/512 = 1.0

    print(f"   RGB aspect ratio: {rgb_aspect:.3f}")
    print(f"   Depth aspect ratio: {depth_aspect:.3f}")

    if abs(depth_aspect - rgb_aspect) < 0.1:
        print("   → Depth has RGB aspect ratio (likely aligned)")
    else:
        print("   → Depth has different aspect ratio (likely raw)")

    # Test 2: Coverage after simple resize
    print("\n3. Coverage test:")
    depth_resized = cv2.resize(depth.astype(np.float32),
                               (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    coverage = (depth_resized > 0).sum() / depth_resized.size * 100
    print(f"   Coverage after simple resize: {coverage:.1f}%")

    if coverage > 50:
        print("   → High coverage (likely aligned)")
    else:
        print("   → Low coverage (likely raw or misaligned)")

    # Test 3: FOV comparison
    print("\n4. FOV analysis:")

    depth_K = np.array(depth_calib['K'])
    fx_d = depth_K[0, 0]
    depth_fov = 2 * np.arctan(depth.shape[1] / 2 / fx_d) * 180 / np.pi

    rgb_K = np.array(rgb_calib['K'])
    fx_r = rgb_K[0, 0]
    rgb_fov = 2 * np.arctan(rgb.shape[1] / 2 / fx_r) * 180 / np.pi

    print(f"   RGB FOV: {rgb_fov:.1f}°")
    print(f"   Depth FOV: {depth_fov:.1f}°")
    print(f"   Difference: {abs(rgb_fov - depth_fov):.1f}°")

    if abs(rgb_fov - depth_fov) > 5:
        print("   → Significant FOV difference (geometric alignment needed)")

    # Test 4: Edge position check
    print("\n5. Edge position analysis:")

    # Find an object edge in depth
    depth_edges = cv2.Canny(cv2.normalize(depth, None, 0, 255,
                            cv2.NORM_MINMAX, cv2.CV_8U), 50, 150)

    # Find edges in RGB
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (depth.shape[1], depth.shape[0]))
    rgb_edges = cv2.Canny(gray_small, 50, 150)

    # Calculate overlap
    overlap = cv2.bitwise_and(depth_edges, rgb_edges)

    if np.sum(depth_edges) > 0:
        overlap_ratio = np.sum(overlap) / np.sum(depth_edges) * 100
        print(f"   Edge overlap (at 512×512): {overlap_ratio:.2f}%")

        if overlap_ratio > 15:
            print("   → Good edge alignment (likely already aligned)")
        elif overlap_ratio > 5:
            print("   → Moderate alignment (check visually)")
        else:
            print("   → Poor alignment (geometric correction needed)")

    # Summary
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Scoring
    score = 0
    if depth_aspect == 1.0 and rgb_aspect > 1.5:
        score -= 1  # Different aspect ratio suggests raw
    if coverage > 70:
        score += 2  # High coverage suggests aligned
    if abs(rgb_fov - depth_fov) > 5:
        score -= 1  # FOV difference suggests raw

    if score >= 1:
        print("✅ LIKELY ALIGNED:")
        print("   Your depth images appear to be pre-aligned to RGB frame.")
        print("   → Use simple resize (cv2.resize)")
        print("   → Geometric alignment NOT needed")
    elif score == 0:
        print("⚠️  UNCERTAIN:")
        print("   Mixed signals. Visual inspection recommended.")
        print("   → Check alignment_check/3_overlay_simple.png")
        print("   → If objects align well, use simple resize")
        print("   → If misaligned, use geometric method")
    else:
        print("❌ LIKELY RAW:")
        print("   Your depth images appear to be raw (not aligned).")
        print("   → Use geometric alignment with extrinsics")
        print("   → Simple resize will be inaccurate")

    print("=" * 80)
    print("\nVisual check: Open 'alignment_check/3_overlay_simple.png'")
    print("If objects align well → Use simple resize")
    print("If objects misalign → Use geometric alignment")
    print("=" * 80)

if __name__ == '__main__':
    test_depth_format()
