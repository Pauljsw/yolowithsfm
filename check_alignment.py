"""
RGB-Depth Alignment Check for Orbbec Femto Bolt

Tests alignment quality by overlaying depth on RGB.
Three methods:
1. Simple resize (naive)
2. With extrinsic transform (correct)
3. Edge comparison (quantitative)
"""
import cv2
import numpy as np
import json
from pathlib import Path


def load_calibration():
    """Load all calibration files."""
    with open('calib/rgb_camera_info.json') as f:
        rgb_calib = json.load(f)

    with open('calib/depth_camera_info.json') as f:
        depth_calib = json.load(f)

    with open('calib/extrinsic_depth_to_color.json') as f:
        extrinsic = json.load(f)

    return rgb_calib, depth_calib, extrinsic


def simple_resize_alignment(rgb, depth):
    """
    Method 1: Simple resize (naive, may have misalignment).

    Just resizes depth to RGB resolution without geometric correction.
    """
    depth_float = depth.astype(np.float32)
    depth_resized = cv2.resize(depth_float, (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    return depth_resized


def geometric_alignment(rgb, depth, rgb_calib, depth_calib, extrinsic):
    """
    Method 2: Geometric alignment using extrinsic transform.

    Properly backprojects depth pixels to 3D, transforms to RGB frame,
    and reprojects to RGB image.
    """
    h_depth, w_depth = depth.shape
    h_rgb, w_rgb = rgb.shape[:2]

    # Depth intrinsics
    K_depth = np.array(depth_calib['K'])
    fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]
    cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]

    # RGB intrinsics
    K_rgb = np.array(rgb_calib['K'])
    fx_r, fy_r = K_rgb[0, 0], K_rgb[1, 1]
    cx_r, cy_r = K_rgb[0, 2], K_rgb[1, 2]

    # Extrinsic transform (Depth → RGB)
    R = np.array(extrinsic['R'])
    t = np.array(extrinsic['t'])  # meters

    # Create output depth map in RGB resolution
    depth_aligned = np.zeros((h_rgb, w_rgb), dtype=np.float32)

    # For each depth pixel
    print("  Transforming depth pixels to RGB frame...")
    v_coords, u_coords = np.meshgrid(range(h_depth), range(w_depth), indexing='ij')
    u_coords = u_coords.flatten()
    v_coords = v_coords.flatten()
    depth_values = depth.flatten() / 1000.0  # mm → meters

    # Valid depth mask
    valid = (depth_values > 0.1) & (depth_values < 10.0)
    u_valid = u_coords[valid]
    v_valid = v_coords[valid]
    z_valid = depth_values[valid]

    # Backproject to 3D (depth camera frame)
    x_depth = (u_valid - cx_d) * z_valid / fx_d
    y_depth = (v_valid - cy_d) * z_valid / fy_d
    points_depth = np.stack([x_depth, y_depth, z_valid], axis=1)  # (N, 3)

    # Transform to RGB camera frame
    points_rgb = (R @ points_depth.T).T + t  # (N, 3)

    # Project to RGB image
    x_rgb_cam = points_rgb[:, 0]
    y_rgb_cam = points_rgb[:, 1]
    z_rgb_cam = points_rgb[:, 2]

    # Valid projection (in front of camera)
    valid_proj = z_rgb_cam > 0
    x_rgb_cam = x_rgb_cam[valid_proj]
    y_rgb_cam = y_rgb_cam[valid_proj]
    z_rgb_cam = z_rgb_cam[valid_proj]

    # Pixel coordinates
    u_rgb = (fx_r * x_rgb_cam / z_rgb_cam + cx_r).astype(int)
    v_rgb = (fy_r * y_rgb_cam / z_rgb_cam + cy_r).astype(int)

    # Valid image coordinates
    valid_coords = (u_rgb >= 0) & (u_rgb < w_rgb) & (v_rgb >= 0) & (v_rgb < h_rgb)
    u_rgb = u_rgb[valid_coords]
    v_rgb = v_rgb[valid_coords]
    z_rgb_mm = (z_rgb_cam[valid_coords] * 1000).astype(np.uint16)

    # Fill depth map (keep minimum depth per pixel)
    for i in range(len(u_rgb)):
        u, v, d = u_rgb[i], v_rgb[i], z_rgb_mm[i]
        if depth_aligned[v, u] == 0 or d < depth_aligned[v, u]:
            depth_aligned[v, u] = d

    print(f"  Aligned pixels: {(depth_aligned > 0).sum():,} / {h_rgb * w_rgb:,} "
          f"({(depth_aligned > 0).sum() / (h_rgb * w_rgb) * 100:.1f}%)")

    return depth_aligned


def visualize_alignment(rgb, depth_aligned, method_name):
    """
    Visualize depth overlay on RGB.
    """
    # Normalize depth for visualization
    depth_valid = depth_aligned.copy()
    depth_valid[depth_valid == 0] = np.nan

    depth_vis = cv2.normalize(depth_aligned, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_colored[depth_aligned == 0] = [0, 0, 0]  # Black for invalid

    # Overlay
    overlay = cv2.addWeighted(rgb, 0.6, depth_colored, 0.4, 0)

    # Add text
    cv2.putText(overlay, method_name, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Coverage stats
    coverage = (depth_aligned > 0).sum() / depth_aligned.size * 100
    cv2.putText(overlay, f"Coverage: {coverage:.1f}%", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return overlay, depth_colored


def edge_comparison(rgb, depth_aligned):
    """
    Quantitative comparison using edge alignment.

    Detects edges in both RGB and depth, computes overlap.
    Good alignment → high edge overlap.
    """
    # RGB edges
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges_rgb = cv2.Canny(gray, 50, 150)

    # Depth edges
    depth_uint8 = cv2.normalize(depth_aligned, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    edges_depth = cv2.Canny(depth_uint8, 30, 100)

    # Valid region (where depth exists)
    valid_mask = (depth_aligned > 0).astype(np.uint8) * 255

    # Edge overlap
    overlap = cv2.bitwise_and(edges_rgb, edges_depth)
    overlap_masked = cv2.bitwise_and(overlap, valid_mask)

    # Metrics
    total_rgb_edges = np.sum(edges_rgb > 0)
    total_depth_edges = np.sum(edges_depth > 0)
    total_overlap = np.sum(overlap_masked > 0)

    if total_depth_edges > 0:
        overlap_ratio = total_overlap / total_depth_edges * 100
    else:
        overlap_ratio = 0

    return edges_rgb, edges_depth, overlap_masked, overlap_ratio


def main():
    print("=" * 80)
    print("RGB-Depth Alignment Check")
    print("=" * 80)

    # Load sample data
    print("\n1. Loading sample data...")
    rgb = cv2.imread('data/rgb/sample.png')
    depth = cv2.imread('data/depth/sample.png', -1)

    if rgb is None or depth is None:
        print("ERROR: Cannot load sample images!")
        print("Make sure 'data/rgb/sample.png' and 'data/depth/sample.png' exist.")
        return

    print(f"  RGB: {rgb.shape}")
    print(f"  Depth: {depth.shape}, range: {depth.min()}-{depth.max()} mm")
    print(f"  Depth coverage: {(depth > 0).sum() / depth.size * 100:.1f}%")

    # Load calibration
    print("\n2. Loading calibration...")
    rgb_calib, depth_calib, extrinsic = load_calibration()
    print(f"  RGB K: fx={rgb_calib['K'][0][0]:.1f}, fy={rgb_calib['K'][1][1]:.1f}")
    print(f"  Depth K: fx={depth_calib['K'][0][0]:.1f}, fy={depth_calib['K'][1][1]:.1f}")
    print(f"  Baseline: {extrinsic['baseline_mm']:.1f} mm")

    # Method 1: Simple resize
    print("\n3. Testing Method 1: Simple Resize...")
    depth_simple = simple_resize_alignment(rgb, depth)
    overlay_simple, depth_vis_simple = visualize_alignment(rgb, depth_simple, "Simple Resize")

    # Method 2: Geometric alignment
    print("\n4. Testing Method 2: Geometric Alignment...")
    depth_geometric = geometric_alignment(rgb, depth, rgb_calib, depth_calib, extrinsic)
    overlay_geometric, depth_vis_geometric = visualize_alignment(
        rgb, depth_geometric, "Geometric (Correct)")

    # Edge comparison
    print("\n5. Quantitative Edge Comparison...")

    print("  Method 1 (Simple):")
    edges_rgb_1, edges_depth_1, overlap_1, ratio_1 = edge_comparison(rgb, depth_simple)
    print(f"    Edge overlap: {ratio_1:.2f}%")

    print("  Method 2 (Geometric):")
    edges_rgb_2, edges_depth_2, overlap_2, ratio_2 = edge_comparison(rgb, depth_geometric)
    print(f"    Edge overlap: {ratio_2:.2f}%")

    # Save results
    print("\n6. Saving results...")
    output_dir = Path('alignment_check')
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / '1_rgb_original.png'), rgb)
    cv2.imwrite(str(output_dir / '2_depth_simple.png'), depth_vis_simple)
    cv2.imwrite(str(output_dir / '3_overlay_simple.png'), overlay_simple)
    cv2.imwrite(str(output_dir / '4_depth_geometric.png'), depth_vis_geometric)
    cv2.imwrite(str(output_dir / '5_overlay_geometric.png'), overlay_geometric)

    # Edge visualization
    edge_vis_simple = np.hstack([edges_rgb_1, edges_depth_1, overlap_1])
    edge_vis_geometric = np.hstack([edges_rgb_2, edges_depth_2, overlap_2])
    cv2.imwrite(str(output_dir / '6_edges_simple.png'), edge_vis_simple)
    cv2.imwrite(str(output_dir / '7_edges_geometric.png'), edge_vis_geometric)

    print(f"  Results saved to: {output_dir}/")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Method 1 (Simple Resize):")
    print(f"  - Coverage: {(depth_simple > 0).sum() / depth_simple.size * 100:.1f}%")
    print(f"  - Edge overlap: {ratio_1:.2f}%")
    print(f"\nMethod 2 (Geometric Alignment):")
    print(f"  - Coverage: {(depth_geometric > 0).sum() / depth_geometric.size * 100:.1f}%")
    print(f"  - Edge overlap: {ratio_2:.2f}%")
    print(f"\nImprovement: {ratio_2 - ratio_1:.2f}% better edge alignment")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if ratio_2 > ratio_1 + 5:
        print("✅ Geometric alignment significantly better!")
        print("   → Use Method 2 (with extrinsic transform) for production")
    elif ratio_2 > ratio_1:
        print("✅ Geometric alignment slightly better")
        print("   → Use Method 2 for best accuracy")
    else:
        print("⚠️  Simple resize comparable to geometric")
        print("   → Method 1 is simpler and may be sufficient")

    print(f"\nVisual inspection: Open '{output_dir}/5_overlay_geometric.png'")
    print("  - Object boundaries should align well")
    print("  - Depth edges should match RGB edges")
    print("=" * 80)


if __name__ == '__main__':
    main()
