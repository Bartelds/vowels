from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool

import os
import numpy as np
import pandas as pd
from dtw import dtw
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS

_cache = {}

def merge_tsv_files(bigdata_file, timing_file):
    # Read the data
    bigdata_df = pd.read_csv(bigdata_file, sep='\t')
    timing_df = pd.read_csv(timing_file, sep='\t')

    # Remove rows containing 0 in any column
    bigdata_df = bigdata_df[~(bigdata_df == 0).any(axis=1)]

    # Merge the two dataframes on 'file' column
    merged_df = pd.merge(bigdata_df, timing_df, on='file')
    
    return merged_df


def bark(freqHz):
    return (( 26.81 / (1 + (1960 / freqHz)) ) - 0.53)


def compute_pairwise_distance(vowel_data):
    distances = []

    for idx1, row1 in vowel_data.iterrows():
        for idx2, row2 in vowel_data.iterrows():
            if idx1 != idx2:  # Skip the same row
                dist = np.sqrt((bark(row1['f1']) - bark(row2['f1']))**2 +
                               (bark(row1['f2']) - bark(row2['f2']))**2)
                distances.append(dist)

    return np.mean(distances)


def compute_distance(df, gender):
    # Group by vowels
    vowel_groups = df.groupby(df['file'].str[-2:])

    # Initialize a DataFrame to store the distances
    distances = pd.DataFrame(index=vowel_groups.groups.keys(), columns=vowel_groups.groups.keys())

    # Calculate the average distance between each pair of vowels
    for name1, group1 in tqdm(vowel_groups):
        for name2, group2 in vowel_groups:
            distances_list = []
            for _, row1 in group1.iterrows():
                for _, row2 in group2.iterrows():
                    if row1['file'] != row2['file']:
                        dist = np.sqrt((bark(row1['f1']) - bark(row2['f1']))**2 + (bark(row1['f2']) - bark(row2['f2']))**2)
                        distances_list.append(dist)
            distances.at[name1, name2] = np.mean(distances_list)

    # save distance matrix
    os.makedirs(f'dist_mats/hillenbrand/formants/', exist_ok=True)
    distances.to_csv(f'dist_mats/hillenbrand/formants/MDS_vowels_{gender}.csv')

    return distances


def visualize_distance_matrix(distance_matrix, gender):
    # Convert the DataFrame to a NumPy array for use with MDS
    flat_distance_matrix = distance_matrix.to_numpy()

    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=998)
    mds_result = mds.fit_transform(flat_distance_matrix)
    
    # Plot the result
    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    plt.scatter(mds_result[:, 0], mds_result[:, 1], edgecolor='w')

    # Annotate the points
    for i, (x, y) in enumerate(zip(mds_result[:, 0], mds_result[:, 1])):
        plt.text(x, y, distance_matrix.index[i], fontsize=10)
        
    plt.title(f'MDS visualization of formant-based vowel distances for {gender} speakers')
    plt.gca().invert_xaxis()
    os.makedirs(f'plots/hillenbrand/formants/', exist_ok=True)
    plt.savefig(f'plots/hillenbrand/formants/MDS_vowels_{gender}.png')
    plt.show()


def main():
    parser = ArgumentParser()
    # Input Data:
    parser.add_argument("-i", "--input_time_path", default="data/PHON/hillenbrand/timedata.tsv")
    parser.add_argument("-g", "--gender", choices=['male', 'woman'], help="Select the gender (options: 'male', 'woman')")
    # Misc:
    parser.add_argument("-p", "--num_procs", default=4, type=int)
    args = parser.parse_args()
    print(args)

    merged_df = merge_tsv_files('data/PHON/hillenbrand/bigdata.tsv', 'data/PHON/hillenbrand/timedata.tsv')
    gender = args.gender
    merged_df = merged_df[merged_df['file'].str.startswith(args.gender[0])]

    distance_matrix = compute_distance(merged_df, gender)
    print(distance_matrix)
    visualize_distance_matrix(distance_matrix, gender)


if __name__ == "__main__":
    main()
