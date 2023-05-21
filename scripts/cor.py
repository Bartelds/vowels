import pandas as pd
import numpy as np

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    # Input Data:
    parser.add_argument("-g", "--gender", choices=['male', 'woman'], help="Select the gender (options: 'male', 'woman')")
    # Model:
    parser.add_argument("-m", "--model", default="wav2vec2-large-960h")
    parser.add_argument("-l", "--layer", default=9, type=int)
    args = parser.parse_args()
    print(args)

    layer_str = str(args.layer).zfill(2)

    # Read the DataFrame from csv
    w2v2 = pd.read_csv(f'dist_mats/hillenbrand/{args.model}/l-{layer_str}/MDS_vowels_{args.gender}.csv', index_col=0)
    formants = pd.read_csv(f'dist_mats/hillenbrand/formants/MDS_vowels_{args.gender}.csv', index_col=0)

    # Normalize the DataFrames
    w2v2 = (w2v2 - np.min(w2v2.values)) / (np.max(w2v2.values) - np.min(w2v2.values))
    formants = (formants - np.min(formants.values)) / (np.max(formants.values) - np.min(formants.values))

    # Flatten the DataFrames
    w2v2_flattened = w2v2.values.flatten()
    formants_flattened = formants.values.flatten()

    # Compute correlation
    correlation = np.corrcoef(w2v2_flattened, formants_flattened)[0, 1]
    print(f"Pearson's correlation between formant-based differences and w2v2 layer {args.layer} based differences of {args.gender} speakers: {round(correlation, 2)}")

if __name__ == "__main__":
    main()
