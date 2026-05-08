import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def load_cd_data(cd_matrix_path, labels_path):
    cd_matrix = np.load(cd_matrix_path)
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return cd_matrix, labels

def colors(labels):
    # Azul: generados, Rojo: referencia
    color = ['blue' if 'generated' in label else 'red' for label in labels]
    return color

def compute_multidimensional_scaling(cd_matrix, labels, colors, save_path='/home/ncaytuir/data-local/PVD_necs/MyScripts/complete_pcs_all/results/mds.png'):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(cd_matrix) # (810, 2)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(embedding[i, 0], embedding[i, 1], color=colors[i], label=label, alpha=0.5)
    plt.title('MDS of Chamfer Distance Matrix')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

cd_matrix_path = '/home/ncaytuir/data-local/PVD_necs/MyScripts/complete_pcs_all/results/cd_matrix.npy'
labels_path = '/home/ncaytuir/data-local/PVD_necs/MyScripts/complete_pcs_all/results/chamfer_distance_labels.txt'

dis_matrix, labels = load_cd_data(cd_matrix_path, labels_path)
color = colors(labels)
compute_multidimensional_scaling(dis_matrix, labels, color)