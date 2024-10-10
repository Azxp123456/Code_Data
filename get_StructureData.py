"""
    Get structure data —— ①Protein distance map
                          ②Secondary structure
"""

import os
import cv2
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
import pickle
from PIL import Image, ImageDraw, ImageFont
from aaindex import AALETTER

DATA_ROOT = '/folder_path'

secondary_Structure_classes = ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'L']     # 8 types of secondary structures
structure_to_index = {structure: index for index, structure in enumerate(secondary_Structure_classes)}  # Structural Index
amino_to_index = {aa: index for index, aa in enumerate(AALETTER)}


def extract_filename(filename):    # ②
    start_index = filename.find('-')

    end_index = filename.find('-', start_index + 1)

    if start_index != -1 and end_index != -1:
        extracted_part = filename[start_index + 1:end_index]
        return extracted_part
    else:
        print("No matching parts found")
        return None

# ①
def DSSP_euclidean_distance():
    """Function: Extract the Cα atomic coordinates of each amino acid residue
        in the DSSP file and calculate the Euclidean distance between each two residues"""

    DSSP_folder_path = "\DSSP_folder_path"
    filelist = os.listdir(DSSP_folder_path)

    total_files = len(filelist)
    processed_files = 0

    # Set the folder path to save the distance files
    save_distance_map_path = "\save_distance_map_path"

    for DSSP_filename in filelist:
        # Actions to handle each file
        file_path = os.path.join(DSSP_folder_path, DSSP_filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        skip_comments = True   # Set the initial state
        data = []    # Store the three-dimensional coordinates of each amino acid residue

        # Automatically find the starting line of the first amino acid residue
        for line in lines:
            if skip_comments:
                if line.startswith('  #'):
                    skip_comments = False
                continue

            # Processing actual data rows
            fields = line.split()

            # Extract the required data, such as X-CA, Y-CA, Z-CA
            # Extract the three-dimensional coordinates of the Cα atom corresponding to each amino acid residue
            x_ca = float(fields[-3])
            y_ca = float(fields[-2])
            z_ca = float(fields[-1])
            data.append([x_ca, y_ca, z_ca])

        data = np.array(data)

        # Calculate the Euclidean distance between every two residues
        num_residues = len(data)
        distance_matrix = np.zeros((num_residues, num_residues))

        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                distance = np.linalg.norm(data[i] - data[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        # print(distance_matrix)

        distance_matrix = np.around(distance_matrix, decimals=1)

        base_filename = extract_filename(DSSP_filename)
        npy_file = os.path.join(save_distance_map_path, f"{base_filename}.npy")
        np.save(npy_file, distance_matrix)
        print(f"The distance matrix has been saved to the {npy_file} file.")

        # 在处理完一个文件后，增加计数器
        processed_files += 1
        print(f"finished: {processed_files}/{total_files}")     # 打印finished语句


def preprocess(image_tensors):
    resized_image_tensors = []
    target_height, target_width = 224, 224

    for image_tensor in image_tensors:  # Image scaling using bilinear interpolation
        resized_image_tensor = tf.image.resize(image_tensor, [target_height, target_width],
                                               method=tf.image.ResizeMethod.BILINEAR)
        resized_image_tensors.append(resized_image_tensor)
    return tf.stack(resized_image_tensors)


def get_SS_Info():
    """
        Read all DSSP files in a folder
        For each DSSP file (each protein sequence)：①Extract (amino acid, secondary structure) tuples
    """
    DSSP_folder_path = "DSSP_folder_path"
    filelist = os.listdir(DSSP_folder_path)
    Entry_Accessions = list()
    AA_SS = list()
    for DSSP_filename in filelist:

        file_path = os.path.join(DSSP_folder_path, DSSP_filename)
        parts_filename = DSSP_filename.split('-')
        SS_list = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            skip_comments = True
            for line in lines:
                if skip_comments:
                    if line.startswith('  #'):
                        skip_comments = False
                    continue

                key_amino_acid = line.split()[3]
                value_secondary_structure = line.split()[4]
                SS_list.append((key_amino_acid, value_secondary_structure))
            Entry_Accessions.append(parts_filename[1])
            AA_SS.append(SS_list)

    df = pd.DataFrame({
        'Entry_Accessions': Entry_Accessions,
        'AA_SS': AA_SS
    })
    print(len(df))

    df.to_pickle(DATA_ROOT + 'SecondStructure.pkl')


def AA_to_OneHot(AA_SS_List):
    """
        One-hot (20-dimensional) vector encoding the amino acids in the protein sequence
    """
    aa_encoded_data = []
    for my_tuple in AA_SS_List:
        num_classes = len(AALETTER)
        encoded_vector = np.zeros(num_classes)
        if my_tuple[0] in amino_to_index:
            index = amino_to_index[my_tuple[0]]
            encoded_vector[index] = 1
        aa_encoded_data.append(encoded_vector)
    aa_encoded_data = np.array(aa_encoded_data)
    return aa_encoded_data


def SS_to_OneHot(AA_SS_List):
    """
        A one-hot (9-dimensional) vector encoding the secondary structure of the amino acids in the protein sequence
    """
    ss_encoded_data = []
    for my_tuple in AA_SS_List:
        num_classes = len(secondary_Structure_classes) + 1
        encoded_vector = np.zeros(num_classes)
        if my_tuple[1] in structure_to_index:
            index = structure_to_index[my_tuple[1]]
            encoded_vector[index] = 1
        else:
            encoded_vector[-1] = 1
        ss_encoded_data.append(encoded_vector)
    ss_encoded_data = np.array(ss_encoded_data)
    return ss_encoded_data


def all_extract_features():
    """
        Get the secondary structure information feature vector of all samples
    """
    ss_df = pd.read_pickle(DATA_ROOT + 'SecondStructure.pkl')
    print('len(ss_df):', len(ss_df))

    # ss_df['SS_feature'] = ss_df['SS_num'].apply(extract_features)
    # ss_df['AA_onehot'] = ss_df['AA_SS'].apply(AA_to_OneHot)
    ss_df['SS_onehot'] = ss_df['AA_SS'].apply(SS_to_OneHot)
    print('len(ss_df):', len(ss_df))

    df2 = pd.read_pickle(DATA_ROOT + 'wheat_swissprot_exp2.pkl')
    print('len(df2):', len(df2))

    df = pd.merge(ss_df, df2, on='Entry_Accessions')
    print('len(df):', len(df))


    df.to_pickle(DATA_ROOT + 'wheat_swissprot_exp3.pkl')


if __name__ == '__main__':

    # DSSP_euclidean_distance()
    # get_SS_Info()
    all_extract_features()