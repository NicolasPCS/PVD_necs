import torch
from pprint import pprint
from metrics.evaluation_metrics import compute_all_metrics
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD

# Nomalización por bounding box
def normalization_bb(pcs):
    # pcs: [B, N, 3]
    bbox_min = pcs.min(dim=1, keepdim=True)[0]
    bbox_max = pcs.max(dim=1, keepdim=True)[0]
    bbox_center = (bbox_max + bbox_min) / 2
    bbox_scale = (bbox_max - bbox_min).max(dim=2, keepdim=True)[0] / 2
    pcs_normalized = (pcs - bbox_center) / bbox_scale
    return pcs_normalized

#samples_path = "/home/ncaytuir/data-local/PVD_necs/output/samples_ivan.pth"
samples_path = "/home/ncaytuir/data-local/PVD_necs/checkpoints/2899/samples.pth"
#ref_path = "/home/ncaytuir/data-local/PVD_necs/checkpoints/2899/reference.pth"
ref_path = "/home/ncaytuir/data-local/PVD_necs/val_data/ref_val_airplane.pt"
batch_size = 50

# Load data

print(f"Loading data: {samples_path} {ref_path}" % ())

sample_data = torch.load(samples_path) # Loads the tensor
sample_pcs = sample_data.contiguous()

ref_data = torch.load(ref_path) # Loads the tensor

if isinstance(ref_data, dict):
    print(f"ref_data is {type(ref_data)}, applying denormalization.")
    ref_pcs = ref_data['ref']
    mean = ref_data['mean'].float()
    std = ref_data['std'].float()

    ref_pcs = ref_pcs * std + mean # Desnormalización

elif torch.is_tensor(ref_data):
    ref_pcs = ref_data.contiguous()

print(sample_pcs.shape)
print(ref_pcs.shape)

# Normalizar ambos conjuntos
#sample_pcs = normalization_bb(sample_pcs.float())
#ref_pcs = normalization_bb(ref_pcs.float())

print(f"Generation sample size: {sample_pcs.size()} reference size: {ref_pcs.size()}")

# Compute metrics
results = compute_all_metrics(sample_pcs, ref_pcs, batch_size)
results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
pprint(f'JSD: {jsd}')
print(f'JSD: {jsd}')
