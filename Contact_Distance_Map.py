import numpy as np
import os
from scipy.ndimage import generic_filter
from scipy.interpolate import griddata
import cv2
from PIL import Image


# Define sliding window size
window_size = 5
target_shape = (224, 224)  # Target size


def process_to_ContactMap():
    DistanceMap_folder_path = "\distance map folder"
    filelist = os.listdir(DistanceMap_folder_path)
    total_files = len(filelist)

    save_contact_map_path = "\contact_map folder"
    threshold = 10.0     # Set the threshold, for example 10 angstroms(Angstroms)
    processed_file = 0
    for distance_map in filelist:
        file_path = os.path.join(DistanceMap_folder_path, distance_map)
        distance_matrix = np.load(file_path)

        adjacency_matrix = np.where(distance_matrix <= threshold, 1, 0)

        base_filename = distance_map.split('.')[0]  # Get protein ID
        contactMap_name = base_filename + "-Contact.npy"
        save_path = os.path.join(save_contact_map_path, contactMap_name)

        np.save(save_path, adjacency_matrix)

        processed_file += 1
        print(f"finished: {processed_file}/{total_files}")


def resize_distance_matrix(distance_matrix, target_shape):
    # Build the target mesh
    x = np.linspace(0, distance_matrix.shape[1] - 1, distance_matrix.shape[1])
    y = np.linspace(0, distance_matrix.shape[0] - 1, distance_matrix.shape[0])
    target_x = np.linspace(0, distance_matrix.shape[1] - 1, target_shape[1])
    target_y = np.linspace(0, distance_matrix.shape[0] - 1, target_shape[0])
    target_xx, target_yy = np.meshgrid(target_x, target_y)

    # Perform bilinear interpolation
    resized_distance_matrix = griddata((x.flatten(), y.flatten()), distance_matrix.flatten(), (target_xx, target_yy),
                                       method='linear')
    return resized_distance_matrix


if __name__ == '__main__':

    process_to_ContactMap()