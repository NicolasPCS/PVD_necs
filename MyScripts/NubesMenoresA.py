import os
import numpy as np

input_path = "/home/ncaytuir/data-local/PVD_necs/ShapeNetCore.v4.PC15k/02691156/train"

umbral_puntos = 6000
cont = 0
archivos_con_pocos_puntos = []

for filename in os.listdir(input_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_path, filename)
        point_cloud = np.load(file_path)

        num_puntos = point_cloud.shape[0]
        if num_puntos < umbral_puntos:
            archivos_con_pocos_puntos.append((filename, num_puntos))
            print(f"[{cont}] Archivo: {filename} - Puntos: {num_puntos}")
            cont += 1

print("\nResumen:")
print(f"Total de archivos con menos de {umbral_puntos} puntos: {len(archivos_con_pocos_puntos)}")
