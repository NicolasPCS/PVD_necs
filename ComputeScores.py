import torch
from pprint import pprint
from metrics.evaluation_metrics import compute_all_metrics
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD

samples_path = "/home/isipiran/PVD_necs/output/test_generation/2025-06-16-12-34-36_airplane/complete_shape_samples_airplane.pt"
ref_path = "/home/isipiran/PVD_necs/val_data/ref_val_airplane.pt"
batch_size = 50

# Load data

print(f"Loading data: {samples_path} {ref_path}" % ())

sample_pcs = torch.load(samples_path) # Loads the tensor
ref_pcs = torch.load(ref_path) # Loads the tensor

sample_pcs = sample_pcs['ref']
ref_pcs = ref_pcs['ref']

print(f"Generation sample size: {sample_pcs.size()} reference size: {ref_pcs.size()}")

# Compute metrics
results = compute_all_metrics(sample_pcs, ref_pcs, batch_size)
results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)
print(results)

jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
pprint(f'JSD: {jsd}')
print(f'JSD: {jsd}')