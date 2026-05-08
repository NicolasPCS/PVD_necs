import os
import numpy as np

input_folder = "/home/ncaytuir/data-local/PVD_necs/ShapeNetCore.v4.PC15k/02691156/train"


def obtain_min_points(input_folder):

    min_points = float("inf")
    min_file = None
    min_cloud = None

    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            filepath = os.path.join(input_folder, filename)
            try:
                point_cloud = np.load(filepath)
                num_points = point_cloud.shape[0]

                if num_points < min_points:
                    min_points = num_points
                    min_file = filename
                    min_cloud = point_cloud

            except Exception as e:
                print(f"Error al cargar {filename}: {e}")
    
    return min_points

# resultados
#print(f"\nArchivo con menos puntos: {min_file}")
print(f"Cantidad de puntos: {obtain_min_points(input_folder)}")
#print("Nube de puntos:")
#print(min_cloud)