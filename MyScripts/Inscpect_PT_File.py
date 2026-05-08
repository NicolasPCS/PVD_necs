import numpy as np
import torch

data = torch.load("/home/ncaytuir/data-local/PVD_necs/val_data/ref_val_chair.pt")

# Ver qué contiene
print(type(data))
#print(data)
#print(data)
#print(data.keys())
print(data['mean'][:1])
#print(data['std'].shape)
print(data['std'][:1])
#print(data)
pc = data.numpy()  # convertir a numpy si aún es tensor

#print("Minimos (x, y, z):", np.min(pc, axis=0))
#print("Maximos (x, y, z):", np.max(pc, axis=0))

if isinstance(data, dict):
    for k, v in data.items():
        print(f"{k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
elif isinstance(data, list):
    print(f"Es una lista con {len(data)} elementos")
    for i, v in enumerate(data[:5]):
        print(f"Elemento {i}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
