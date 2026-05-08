"""
Modified from the official PVD implementation:
https://github.com/alexzhou907/PVD
Zhou et al., "Point-Voxel Diffusion for 3D Shape Generation", arXiv:2104.03670
"""

import torch
from pprint import pprint
from metrics.evaluation_metrics import compute_all_metrics
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
import argparse
import json
from pathlib import Path

"""
Helpers
"""

# Bounding box normalization
def normalization_bb(pcs, eps=1e-8):
    bbox_min = pcs.min(dim=1, keepdim=True).values
    bbox_max = pcs.max(dim=1, keepdim=True).values
    bbox_center = (bbox_max + bbox_min) / 2
    bbox_scale = (bbox_max - bbox_min).max(dim=2, keepdim=True).values / 2
    return (pcs - bbox_center) / (bbox_scale + eps)

def is_in_minus1_1(pcs, tol=0.05, min_radius=0.75, max_radius=1.05):
    bbox_min = pcs.min(dim=1).values
    bbox_max = pcs.max(dim=1).values
    radius = ((bbox_max - bbox_min).max(dim=1).values) / 2
    mean_radius = radius.mean().item()

    min_v = pcs.min().item()
    max_v = pcs.max().item()

    in_range = min_v >= -1.0 - tol and max_v <= 1.0 + tol
    good_radius = min_radius <= mean_radius <= max_radius

    print(f"mean_bbox_radius={mean_radius:.6f}")

    return in_range and good_radius

def ensure_minus1_1(pcs, name, req_norm=False, tol=0.05):
    print(f"\n[CHECK SCALE] {name}")
    print(
        f"Before: shape={tuple(pcs.shape)}, "
        f"min={pcs.min().item():.6f}, "
        f"max={pcs.max().item():.6f}, "
        f"mean={pcs.mean().item():.6f}, "
        f"std={pcs.std().item():.6f}"
    )

    if not req_norm:
        print(f"[INFO] {name}: req_norm=False -> se mantiene escala original.")
        return pcs.float()

    if is_in_minus1_1(pcs, tol=tol):
        print(f"[OK] {name}: ya está aproximadamente en [-1,1]. No se normaliza.")
        return pcs.float()

    print(f"[AUTO] {name}: no está en [-1,1]. Se normaliza con bounding box.")
    pcs = normalization_bb(pcs.float())

    print(
        f"After: min={pcs.min().item():.6f}, "
        f"max={pcs.max().item():.6f}, "
        f"mean={pcs.mean().item():.6f}, "
        f"std={pcs.std().item():.6f}"
    )

    return pcs

def stats(name, pcs):
    print(
        f"{name}: "
        f"mean={pcs.mean().item():.4f}, "
        f"std={pcs.std().item():.4f}, "
        f"min={pcs.min().item():.4f}, "
        f"max={pcs.max().item():.4f}"
    )

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--sample_pth', type=str, default='', required=True)
ap.add_argument('-r', '--reference_pth', type=str, default='', required=True)
ap.add_argument('-o', '--out_pth', type=str, default='results_metrics_symmetry.json', required=False)
ap.add_argument('-bs', '--batch_size', type=int, default=50, required=False)
ap.add_argument('-n', '--req_norm', action='store_true', help='Si se activa, normaliza automáticamente a [-1,1] solo las nubes que no estén ya en ese rango.')

args = ap.parse_args()

samples_path = args.sample_pth
ref_path = args.reference_pth
output_path = Path(args.out_pth)
batch_size = args.batch_size

# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("samples_path:", args.sample_pth)
print("ref_path:", args.reference_pth)
print("output_path:", args.out_pth)
print("batch_size:", args.batch_size)
print("req_norm:", args.req_norm)

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
else:
    raise ValueError(f"Unsupported reference format: {type(ref_data)}")

# Shape checks
assert sample_pcs.ndim == 3, f"sample_pcs debe ser [B,N,3], pero tiene {sample_pcs.shape}"
assert ref_pcs.ndim == 3, f"ref_pcs debe ser [B,N,3], pero tiene {ref_pcs.shape}"
assert sample_pcs.shape[-1] == 3, f"sample_pcs última dimensión debe ser 3, pero tiene {sample_pcs.shape}"
assert ref_pcs.shape[-1] == 3, f"ref_pcs última dimensión debe ser 3, pero tiene {ref_pcs.shape}"

sample_pcs = sample_pcs.float().to(device)
ref_pcs = ref_pcs.float().to(device)

print("\n[LOADED DATA]")
stats("samples loaded", sample_pcs)
stats("refs loaded", ref_pcs)

# Auto-scale alignment only if --req_norm is used
sample_pcs = normalization_bb(sample_pcs)
ref_pcs = normalization_bb(ref_pcs)

""" sample_pcs = ensure_minus1_1(
    sample_pcs,
    name="samples",
    req_norm=args.req_norm
)

ref_pcs = ensure_minus1_1(
    ref_pcs,
    name="references",
    req_norm=args.req_norm
) """

print("\n[FINAL SCALE CHECK]")
stats("samples final", sample_pcs)
stats("refs final", ref_pcs)

""" if args.req_norm:
    assert sample_pcs.min() >= -1.1 and sample_pcs.max() <= 1.1, "Samples fuera de escala [-1,1]"
    assert ref_pcs.min() >= -1.1 and ref_pcs.max() <= 1.1, "References fuera de escala [-1,1]"

print(f"\nGeneration sample size: {sample_pcs.size()} reference size: {ref_pcs.size()}") """

# Compute metrics
results = compute_all_metrics(sample_pcs, ref_pcs, batch_size)
results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

# ---- Append to JSON ----
output_path = Path(args.out_pth)

# Añadir metadata
results["sample_pth"] = args.sample_pth

if output_path.exists():
    with output_path.open("r") as f:
        all_results = json.load(f)
else:
    all_results = []

all_results.append(results)

with output_path.open("w") as f:
    json.dump(all_results, f, indent=4)

print(f"[OK] Appended results to {output_path.resolve()}")

""" # Normalice both sets
if args.req_norm:
    sample_pcs = normalization_bb(sample_pcs.float())
    ref_pcs = normalization_bb(ref_pcs.float())
else:
    sample_pcs = sample_pcs.float()
    ref_pcs = ref_pcs.float()

stats("samples", sample_pcs)
stats("refs", ref_pcs)

print(f"Generation sample size: {sample_pcs.size()} reference size: {ref_pcs.size()}")

# Compute metrics
results = compute_all_metrics(sample_pcs, ref_pcs, batch_size)
results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

# ---- Append to JSON ----
if output_path.exists():
    with output_path.open("r") as f:
        all_results = json.load(f)
else:
    all_results = []

all_results.append(results)

with output_path.open("w") as f:
    json.dump(all_results, f, indent=4)

print(f"[OK] Appended results to {output_path.resolve()}") """

#jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
#pprint(f'JSD: {jsd}')
#print(f'JSD: {jsd}')

"""
PVD

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/airplane/ckpt_6199/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/car/ckpt_3299/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/chair/ckpt_1199/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm False

LION

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Airplane/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Car/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Chair/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm False

XCube

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_airplane_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_car_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_chair_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

SLIDE 3D

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_airplane_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_car_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_chair_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm False

"""