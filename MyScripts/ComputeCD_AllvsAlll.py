import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import os
import json

# More scalable to thousand of points
def chamfer_distance(pc1, pc2):
    try:
        tree = KDTree(pc2)
        dist_point_cloud1 = tree.query(pc1)[0]
        tree = KDTree(pc1)
        dist_point_cloud2 = tree.query(pc2)[0]

        chamfer = np.mean(dist_point_cloud1) + np.mean(dist_point_cloud2)

        return chamfer
    except:
        raise Exception("Error while trying to compute chamfer distance.")
    
def compute_chamfer_distance_all_vs_all(complete_pcs_all_path, results_path):

    # Load all point clouds
    all_pcs = []
    labels = []

    pc_files = sorted([f for f in os.listdir(complete_pcs_all_path) if f.endswith(".npy")])

    for f in pc_files:
        labels.append(f.replace('.npy', ''))
        pc = np.load(os.path.join(complete_pcs_all_path, f))
        all_pcs.append(pc)
    
    n = len(all_pcs)
    cd_matrix = np.zeros((n, n))
    cd_json = {}

    for i in tqdm(range(n), desc="Computing Chamfer Distance"):
        label_i = labels[i]
        cd_json[label_i] = {}

        for j in range(n):
            label_j = labels[j]
            cd = chamfer_distance(all_pcs[i], all_pcs[j])
            cd_matrix[i, j] = cd
            cd_json[label_i][label_j] = float(cd)  # Convert to float for JSON serialization
        
    # Save matrix
    np.save(os.path.join(results_path, "cd_matrix.npy"), cd_matrix)

    # Save labels
    with open(os.path.join(results_path, 'chamfer_distance_labels.txt'), 'w') as f:
        for label in labels:
            f.write(label + '\n')

    with open(os.path.join(results_path, "cd_computation.json"), 'w') as f:
        json.dump(cd_json, f)

    return

complete_pcs_all_path = '/home/ncaytuir/data-local/PVD_necs/MyScripts/complete_pcs_all'
results_path = '/home/ncaytuir/data-local/PVD_necs/MyScripts/complete_pcs_all/results'
compute_chamfer_distance_all_vs_all(complete_pcs_all_path, results_path)