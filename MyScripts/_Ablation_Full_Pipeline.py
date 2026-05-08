import os
import json
import torch
import numpy as np
import open3d as o3d
import argparse

"""Helpers"""

# Compute FPS
def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

# Argument parser
parser = argparse.ArgumentParser(description="Create PKL files from PC data")
parser.add_argument("input_path", type=str, help="Path to the input point cloud directory")
parser.add_argument("output_path_to_save_pth", type=str, help="Path to the output directory")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path_to_save_pth

cont = 0
num_points = 2048

mirrored_halfs = []

for filename in os.listdir(input_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_path, filename)
        point_cloud = np.load(file_path)

        # Compute the centroid of the bounding
        bbox_min = np.min(point_cloud, axis=0)
        bbox_max = np.max(point_cloud, axis=0)
        bbox_center = (bbox_max + bbox_min) / 2.0

        # Move cloud to x = 0
        normalized_pc = point_cloud - bbox_center

        # Split both sides according to plane x = 0
        half_cloud_right = normalized_pc[normalized_pc[:, 0] >= 0]
        half_cloud_left = normalized_pc[normalized_pc[:, 0] <= 0]

        # Select the side with more points
        if half_cloud_right.shape[0] >= half_cloud_left.shape[0]:
            selected_half = half_cloud_right.copy()
            selected_side = "right"
        else:
            selected_half = half_cloud_left.copy()
            selected_side = "left"

        if selected_half.shape[0] == 0:
            print(f"Skipping {filename}: empty selected half.")
            continue

        # Mirror selected half with respect to x = 0
        mirrored_half = selected_half.copy()
        mirrored_half[:, 0] *= -1

        # Build full symmetric object
        full_points = np.concatenate([selected_half, mirrored_half], axis=0)

        full_points = farthest_point_sampling(full_points, num_points)

        mirrored_halfs.append(full_points)

if len(mirrored_halfs) == 0:
    raise RuntimeError(f"No valid .npy point clouds were found in: {input_path}")

all_pcs = np.stack(mirrored_halfs, axis=0).astype(np.float32)

print("all_pcs.shape:", all_pcs.shape)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save as torch tensor
all_pcs_tensor = torch.from_numpy(all_pcs)
torch.save(all_pcs_tensor, output_path)

print(f"Saved successfully to: {output_path}")