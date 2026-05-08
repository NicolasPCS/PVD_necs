import numpy as np
import open3d as o3d
import os
import torch

# Compute FPS
def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

input_dir = "/home/ncaytuir/data-local/PVD_necs/output/test_generation/2025-06-19-17-36-25/syn/half_pcs"
output_dir = "/home/ncaytuir/data-local/PVD_necs/output/test_generation/2025-06-19-17-36-25/syn/complete_pcs"

files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
cont = 0
num_points = 2048

for fname in files:
    input_path = os.path.join(input_dir, fname)

    # Cargar la nube
    points = np.load(input_path)

    if points.shape[0] < 2048:
        print(fname)
        print(points.shape[0])
        raise("Points are less than 2048")

    # Mover todo a x > 0
    shift = abs(np.min(points[:, 0])) - 0.03
    points[:, 0] += shift

    # Reflejar la nube
    points_mirrored = points.copy()
    points_mirrored[:, 0] *= -1

    # Unir con la nube original
    full_points = np.concatenate([points, points_mirrored], axis=0)

    sampled_points = farthest_point_sampling(full_points, num_points)

    print(f"Antes {full_points.shape} despues {sampled_points.shape}")

    # Normalization
    # Calcula la media del bounding box
    """ bbox_min = np.min(sampled_points, axis=0)
    bbox_max = np.max(sampled_points, axis=0)
    bbox_center = (bbox_max + bbox_min) / 2

    # Calcular la escala
    bbox_scale = np.max(bbox_max - bbox_min) / 2

    # Normalizar la nube de puntos
    nomalized_pc = (sampled_points - bbox_center) / bbox_scale """
    
    # Guardar la nueva nube de puntos
    output_file = os.path.join(output_dir, fname)
    np.save(output_file, sampled_points)

    print(f"Guaradada {cont}")
    cont += 1