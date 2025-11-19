"""
Export Crack Groups as 3D Meshes for CloudCompare

Creates visible mesh surfaces instead of just points:
1. Bounding box meshes for each group (simple)
2. Actual mask polygon meshes (detailed)
"""

import numpy as np
import json
import cv2
from pathlib import Path
import logging
from typing import List, Dict, Tuple

from src.colmap_io import read_images_binary, qvec2rotmat

logger = logging.getLogger(__name__)


class MeshExporter:
    """Export crack groups as 3D meshes"""

    def __init__(
        self,
        rgb_calib_path: str,
        depth_calib_path: str,
        colmap_model_path: str,
        depth_scale: float = 1000.0
    ):
        # Load calibration
        with open(rgb_calib_path) as f:
            rgb_calib = json.load(f)
        with open(depth_calib_path) as f:
            depth_calib = json.load(f)

        self.K_rgb = np.array(rgb_calib['K'])
        self.rgb_size = (rgb_calib['width'], rgb_calib['height'])
        self.K_depth = np.array(depth_calib['K'])
        self.depth_size = (depth_calib['width'], depth_calib['height'])
        self.depth_scale = depth_scale

        # Load COLMAP
        self.colmap_images = read_images_binary(
            Path(colmap_model_path) / "images.bin"
        )
        logger.info(f"Loaded {len(self.colmap_images)} camera poses")

    def backproject_mask_to_mesh(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        camera_pose: Dict,
        sample_rate: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backproject mask to 3D mesh.

        Args:
            mask: (H, W) boolean mask
            depth: (H_d, W_d) depth image
            camera_pose: {"R": (3,3), "t": (3,)}
            sample_rate: Subsample factor (5 = every 5th pixel)

        Returns:
            vertices: (N, 3) 3D vertices
            faces: (M, 3) triangle faces (indices into vertices)
        """
        H, W = mask.shape
        H_d, W_d = self.depth_size

        # Get mask pixel coordinates
        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        # Subsample for performance
        if sample_rate > 1:
            indices = np.arange(0, len(x_coords), sample_rate)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]

        # Scale to depth image
        x_depth = (x_coords * W_d / W).astype(int)
        y_depth = (y_coords * H_d / H).astype(int)

        # Clip bounds
        valid = (
            (x_depth >= 0) & (x_depth < W_d) &
            (y_depth >= 0) & (y_depth < H_d)
        )
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        x_depth = x_depth[valid]
        y_depth = y_depth[valid]

        # Get depth
        depth_values = depth[y_depth, x_depth].astype(np.float32) / self.depth_scale
        valid_depth = (depth_values > 0.1) & (depth_values < 10.0)

        x_coords = x_coords[valid_depth]
        y_coords = y_coords[valid_depth]
        depth_values = depth_values[valid_depth]

        if len(depth_values) < 3:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        # Backproject to 3D
        fx, fy = self.K_rgb[0, 0], self.K_rgb[1, 1]
        cx, cy = self.K_rgb[0, 2], self.K_rgb[1, 2]

        x_cam = (x_coords - cx) * depth_values / fx
        y_cam = (y_coords - cy) * depth_values / fy
        z_cam = depth_values

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to global
        R = camera_pose["R"]
        t = camera_pose["t"]
        vertices = (R.T @ (points_cam - t).T).T

        # Create simple triangulation (Delaunay would be better but complex)
        # For now, create a fan triangulation from centroid
        if len(vertices) < 3:
            return vertices, np.zeros((0, 3), dtype=int)

        # Simple fan triangulation
        faces = []
        centroid_idx = len(vertices)
        centroid = vertices.mean(axis=0)
        vertices = np.vstack([vertices, centroid.reshape(1, 3)])

        for i in range(len(vertices) - 2):
            faces.append([i, i + 1, centroid_idx])

        # Close the fan
        faces.append([len(vertices) - 2, 0, centroid_idx])

        return vertices, np.array(faces, dtype=int)

    def create_bbox_mesh(self, center: np.ndarray, extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create bounding box mesh.

        Args:
            center: (3,) center point
            extent: (3,) box size [dx, dy, dz]

        Returns:
            vertices: (8, 3) box vertices
            faces: (12, 3) triangles (2 per face)
        """
        dx, dy, dz = extent / 2

        # 8 vertices
        vertices = np.array([
            [center[0] - dx, center[1] - dy, center[2] - dz],  # 0
            [center[0] + dx, center[1] - dy, center[2] - dz],  # 1
            [center[0] + dx, center[1] + dy, center[2] - dz],  # 2
            [center[0] - dx, center[1] + dy, center[2] - dz],  # 3
            [center[0] - dx, center[1] - dy, center[2] + dz],  # 4
            [center[0] + dx, center[1] - dy, center[2] + dz],  # 5
            [center[0] + dx, center[1] + dy, center[2] + dz],  # 6
            [center[0] - dx, center[1] + dy, center[2] + dz],  # 7
        ])

        # 12 triangles (2 per face)
        faces = np.array([
            # Bottom
            [0, 1, 2], [0, 2, 3],
            # Top
            [4, 6, 5], [4, 7, 6],
            # Front
            [0, 5, 1], [0, 4, 5],
            # Back
            [2, 7, 3], [2, 6, 7],
            # Left
            [0, 7, 4], [0, 3, 7],
            # Right
            [1, 6, 2], [1, 5, 6],
        ], dtype=int)

        return vertices, faces

    def export_group_meshes(
        self,
        groups: List[Dict],
        masks_3d_dir: str,
        rgb_dir: str,
        depth_dir: str,
        output_path: str,
        mode: str = "bbox"  # "bbox" or "detailed"
    ):
        """
        Export all groups as meshes to PLY.

        Args:
            groups: List of group dicts
            masks_3d_dir: Directory with mask 3D data
            rgb_dir: RGB images directory
            depth_dir: Depth images directory
            output_path: Output PLY file
            mode: "bbox" for bounding boxes, "detailed" for actual mask meshes
        """
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        # Color palette
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab20')

        for group_idx, group in enumerate(groups):
            color = np.array(cmap(group_idx / len(groups))[:3]) * 255
            color = color.astype(np.uint8)

            if mode == "bbox":
                # Simple bounding box
                center = np.array(group['centroid_3d'])
                extent = np.array(group['extent_3d'])

                vertices, faces = self.create_bbox_mesh(center, extent)
                colors = np.tile(color, (len(vertices), 1))

            elif mode == "detailed":
                # Detailed mask meshes
                vertices_list = []
                faces_list = []

                for mask_info in group['masks']:
                    image_name = mask_info['image']
                    mask_idx = mask_info['mask_idx']

                    # Load mask 3D data
                    mask_3d_path = Path(masks_3d_dir) / f"{Path(image_name).stem}.json"
                    if not mask_3d_path.exists():
                        continue

                    # Load original data
                    rgb_path = Path(rgb_dir) / image_name
                    timestamp = image_name.replace("camera_RGB_", "").replace(".png", "")
                    depth_path = Path(depth_dir) / f"camera_DPT_{timestamp}.png"

                    if not rgb_path.exists() or not depth_path.exists():
                        continue

                    # Load YOLO mask
                    yolo_path = Path(f"data/yolo_masks/{image_name}").with_suffix('.json')
                    if not yolo_path.exists():
                        continue

                    with open(yolo_path) as f:
                        yolo_data = json.load(f)

                    if 'masks' in yolo_data:
                        detections = yolo_data['masks']
                    else:
                        detections = yolo_data

                    if mask_idx >= len(detections):
                        continue

                    detection = detections[mask_idx]

                    # Reconstruct mask
                    img = cv2.imread(str(rgb_path))
                    H, W = img.shape[:2]

                    if 'polygon' in detection:
                        polygon = np.array(detection['polygon'], dtype=np.int32)
                    else:
                        continue

                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask, [polygon], 1)
                    mask = mask.astype(bool)

                    # Load depth
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

                    # Get camera pose
                    colmap_image = None
                    for img_id, img_data in self.colmap_images.items():
                        if img_data.name == image_name:
                            colmap_image = img_data
                            break

                    if colmap_image is None:
                        continue

                    R_w2c = qvec2rotmat(colmap_image.qvec)
                    t_w2c = colmap_image.tvec
                    camera_pose = {
                        "R": R_w2c.T,
                        "t": -R_w2c.T @ t_w2c
                    }

                    # Backproject to mesh
                    v, f = self.backproject_mask_to_mesh(
                        mask, depth, camera_pose, sample_rate=10
                    )

                    if len(v) > 0:
                        vertices_list.append(v)
                        faces_list.append(f + len(all_vertices) + sum(len(vv) for vv in vertices_list[:-1]))

                if len(vertices_list) > 0:
                    vertices = np.vstack(vertices_list)
                    faces = np.vstack(faces_list) if len(faces_list) > 0 else np.zeros((0, 3), dtype=int)
                    colors = np.tile(color, (len(vertices), 1))
                else:
                    continue
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Accumulate
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            all_colors.append(colors)
            vertex_offset += len(vertices)

            logger.info(f"Group {group_idx + 1}/{len(groups)}: {len(vertices)} vertices, {len(faces)} faces")

        # Combine all
        if len(all_vertices) == 0:
            logger.error("No meshes generated!")
            return

        all_vertices = np.vstack(all_vertices)
        all_faces = np.vstack(all_faces)
        all_colors = np.vstack(all_colors)

        # Write PLY with faces
        write_ply_mesh(output_path, all_vertices, all_faces, all_colors)
        logger.info(f"Exported {len(all_vertices)} vertices, {len(all_faces)} faces to {output_path}")


def write_ply_mesh(filepath: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """
    Write mesh to PLY file.

    Args:
        filepath: Output path
        vertices: (N, 3) vertices
        faces: (M, 3) triangle indices
        colors: (N, 3) RGB colors (0-255)
    """
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        for vertex, color in zip(vertices, colors):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} "
                   f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

        # Faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export crack groups as 3D meshes for CloudCompare"
    )
    parser.add_argument('--groups', required=True, help='groups.json file')
    parser.add_argument('--masks-3d-dir', required=True, help='Directory with mask 3D data')
    parser.add_argument('--rgb-dir', required=True, help='RGB images directory')
    parser.add_argument('--depth-dir', required=True, help='Depth images directory')
    parser.add_argument('--rgb-calib', required=True)
    parser.add_argument('--depth-calib', required=True)
    parser.add_argument('--colmap-model', required=True)
    parser.add_argument('--output', required=True, help='Output PLY file')
    parser.add_argument(
        '--mode', choices=['bbox', 'detailed'], default='bbox',
        help='bbox: simple bounding boxes (fast), detailed: actual mask meshes (slow)'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Load groups
    with open(args.groups) as f:
        groups = json.load(f)

    # Create exporter
    exporter = MeshExporter(
        rgb_calib_path=args.rgb_calib,
        depth_calib_path=args.depth_calib,
        colmap_model_path=args.colmap_model
    )

    # Export
    exporter.export_group_meshes(
        groups=groups,
        masks_3d_dir=args.masks_3d_dir,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        output_path=args.output,
        mode=args.mode
    )

    print("\n" + "="*60)
    print("MESH EXPORT COMPLETE")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    print("\nOpen in CloudCompare:")
    print(f"  cloudcompare {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
