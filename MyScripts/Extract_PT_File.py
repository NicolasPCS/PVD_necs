import os
import torch
import numpy as np
#import polyscope as ps

data = torch.load("/home/ncaytuir/data-local/PVD_necs/output/test_generation_new2/2025-06-22-20-39-35/syn/reference.pth", map_location='cpu').contiguous()
output_dir = "/home/ncaytuir/data-local/PVD_necs/MyScripts/complete_pcs_reference"

j = 0

for i in range(data.shape[0]):
    np.save(os.path.join(output_dir, f"reference_pc_{i}.npy"), data[i].numpy())
    j += 1

print("Done: ", j)