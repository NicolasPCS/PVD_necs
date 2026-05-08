import os
import numpy as np

input_path = "/home/ncaytuir/data-local/PVD_necs/ShapeNetCore.v4.PC15k/02691156/train"
umbral_puntos = 6000
cont = 0

for filename in os.listdir(input_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_path, filename)
        try:
            point_cloud = np.load(file_path)
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
            continue

        num_puntos = point_cloud.shape[0]
        if num_puntos < umbral_puntos:
            os.remove(file_path)
            print(f"[{cont}] Archivo eliminado: {filename} - Puntos: {num_puntos}")
            cont += 1

print(f"\nTotal de archivos eliminados: {cont}")
