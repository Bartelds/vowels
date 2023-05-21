from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple

import os
import numpy as np
import pandas as pd
from dtw import dtw
from scipy.stats import pearsonr
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


def load_features(time_path,
                  feat_path,
                  rate,
                  featurizer_fn) -> List[Tuple[str, np.ndarray]]:

    if str(feat_path).endswith(".npy"):
        feats = np.load(feat_path)
    else:
        if "featurizer" not in _cache:
            _cache["featurizer"] = featurizer_fn()
        feats = _cache["featurizer"](feat_path)

    token_feats = []
    with open(time_path) as f:
        next(f)
        for line in f:
            file, start, end, _, _ = line.rstrip().split("\t")
            start = round((float(start) / 1000) * rate)
            end = round((float(end) / 1000) * rate)
            if feat_path.split('/')[-2] == file:
                token_feats.append((file, feats[start:end]))

    return token_feats


def compute_distance(features, model, layer_str, gender):
    # Initialize a dictionary to store the arrays for each vowel
    vowel_arrays = {}

    # Traverse each sublist
    for sublist in features:
        # Get the vowel (last two characters of the string)
        vowel = sublist[0][0][-2:]
        file = sublist[0][0]

        # Add the array to the corresponding vowel in the dictionary
        if vowel not in vowel_arrays:
            vowel_arrays[vowel] = []
        vowel_arrays[vowel].append((file, sublist[0][1]))

    # Initialize a matrix to store the distance between each pair of vowels
    distance_matrix = {}

    # Calculate the distance between each pair of vowels
    for vowel1 in tqdm(vowel_arrays):
        for vowel2 in vowel_arrays:
            distances = []

            # calculate dtw distance for each pair of features across speakers
            for feature1 in vowel_arrays[vowel1]:
                for feature2 in vowel_arrays[vowel2]:
                    if feature1[0] != feature2[0]:
                        dobj = dtw(feature1[1], feature2[1])
                        distances.append(dobj.normalizedDistance)

            # store the average distance in the matrix
            distance_matrix[(vowel1, vowel2)] = np.mean(distances)

    # Convert dictionary to DataFrame
    index = pd.MultiIndex.from_tuples(distance_matrix.keys())
    distance_matrix_df = pd.DataFrame(list(distance_matrix.values()), index=index)

    # Pivot DataFrame to get the desired format
    distance_matrix_df = distance_matrix_df.unstack()

    # Reset the column names
    distance_matrix_df.columns = distance_matrix_df.columns.droplevel(0)

    # save distance matrix
    os.makedirs(f'dist_mats/hillenbrand/{model}/l-{layer_str}/', exist_ok=True)
    distance_matrix_df.to_csv(f'dist_mats/hillenbrand/{model}/l-{layer_str}/MDS_vowels_{gender}.csv')

    return distance_matrix_df


def visualize_distance_matrix(distance_matrix, model, layer_str, gender):
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
        
    plt.title(f'MDS visualization of vowel distances for {gender} speakers using {model} l{layer_str}')
    # plt.gca().invert_xaxis()
    os.makedirs(f'plots/hillenbrand/{model}/l-{layer_str}/', exist_ok=True)
    plt.savefig(f'plots/hillenbrand/{model}/l-{layer_str}/MDS_vowels_{gender}.png')
    plt.show()


def main():
    parser = ArgumentParser()
    # Input Data:
    parser.add_argument("-a", "--input_feat_path", default="feats/{model}/{speaker}/layer-{layer}.npy")
    parser.add_argument("-i", "--input_time_path", default="data/PHON/hillenbrand/timedata.tsv")
    parser.add_argument("-g", "--gender", choices=['male', 'woman'], help="Select the gender (options: 'male', 'woman')")
    # Model:
    parser.add_argument("-m", "--model", default="wav2vec2-large-960h")
    parser.add_argument("-l", "--layer", default=9, type=int)
    parser.add_argument("--rate", default=50, type=int)
    # Misc:
    parser.add_argument("-p", "--num_procs", default=4, type=int)
    args = parser.parse_args()
    print(args)

    layer_str = str(args.layer).zfill(2)
    merged_df = merge_tsv_files('data/PHON/hillenbrand/bigdata.tsv', 'data/PHON/hillenbrand/timedata.tsv')

    def featurizer_fn():
        from extract_feats import load_wav2vec2_featurizer
        return load_wav2vec2_featurizer(args.model, args.layer)

    # Determine input/output paths
    input_feat_path = args.input_feat_path.format(
        model=args.model,
        layer=layer_str,
        speaker="{speaker}",
    )

    gender = args.gender

    def load_features_fn(speaker):
        return load_features(
            args.input_time_path.format(speaker=speaker),
            input_feat_path.format(speaker=speaker),
            featurizer_fn=featurizer_fn,
            rate=args.rate
        )
    
    features = [load_features_fn(speaker) for speaker in merged_df['file'].to_list() if speaker[0] == gender[0]]

    distance_matrix = compute_distance(features, args.model, layer_str, gender)
    print(distance_matrix)
    visualize_distance_matrix(distance_matrix, args.model, layer_str, gender)


if __name__ == "__main__":
    main()
