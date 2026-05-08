import os
import sys
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from test_generation import get_dataset

input_path = "/home/ncaytuir/data-local/PVD_necs/output/test_generation/2025-06-19-17-36-25/syn/complete_pcs"
output_path_samples = "/home/ncaytuir/data-local/PVD_necs/output/test_generation/2025-06-19-17-36-25/syn/created_pth/samples.pth"
output_path_reference = "/home/ncaytuir/data-local/PVD_necs/output/test_generation/2025-06-19-17-36-25/syn/created_pth/reference.pth"

# Additional
dataroot = '/home/ncaytuir/data-local/PVD_necs/ShapeNetCore.v2.PC15k/'
category = 'airplane'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute FPS
def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

# Listar y ordenar archivos .npy
file_list = sorted([f for f in os.listdir(input_path) if f.endswith(".npy")]) #[:250] # 250 could be adjusted

all_pcs = []
num_points = 2048

for filename in file_list:
    file_path = os.path.join(input_path, filename)
    pc = np.load(file_path)

    sampled_points = farthest_point_sampling(pc, num_points)

    print(f"La nube tenia {pc.shape[0]} ahora tiene {sampled_points.shape[0]}")

    all_pcs.append(sampled_points)

all_pcs = np.array(all_pcs) # (405, 2048, 3)
print(all_pcs.shape)

_, test_dataset = get_dataset(dataroot, num_points, category)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=16, drop_last=False)

with torch.no_grad():
    samples = []
    ref = []

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Importing test data"):
        print(data)
        x = data['test_points'].transpose(1,2).to(device) # [B, 3, 2048]
        m = data['mean'].float().to(device)
        s = data['std'].float().to(device)
        print(m)
        print(s)

        x = x.transpose(1,2).contiguous() # [B, 2048, 3]

        B = x.shape[0]

        for j in range(B):
            pc_i = torch.tensor(all_pcs[len(samples)]).float().to(device) # len(samples): indice global

            pc_i = pc_i*s[j] + m[j]
            x_j = x[j]*s[j] + m[j]

            samples.append(pc_i.cpu())
            ref.append(x_j.cpu())

        """ pc_i = all_pcs[i]
        pc_i = torch.tensor(pc_i).float().to(s.device)

        pc_i = pc_i * s + m
        x = x * s + m 
        
        samples.append(pc_i)
        ref.append(x) """

    
    samples = torch.stack(samples, dim=0)
    ref = torch.stack(ref, dim=0)

    torch.save(samples, output_path_samples)
    torch.save(ref, output_path_reference)