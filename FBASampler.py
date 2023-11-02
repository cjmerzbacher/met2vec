from cobra.io import read_sbml_model
from cobra.sampling import sample
from tqdm import tqdm

import logging
import argparse
import os

logging.getLogger('cobra').setLevel(logging.CRITICAL)

MODELS_LOCATION = 'gems'

jp = os.path.join
ls = os.listdir

def get_args():
    parser = argparse.ArgumentParser("FBA Sampler")
    parser.add_argument("folder", help='The folder being sampled from.')
    parser.add_argument("-n", type=int, default=100, help='Number of samples to be made per file.')
    parser.add_argument("-m", "--method", default='optgp', choices=['optgp', 'achr'], help='Sampling method to be used.')
    return parser.parse_args()

def main():
    args = get_args()

    models_folder = jp(args.folder, MODELS_LOCATION)
    files = [jp(models_folder, f) for f in ls(models_folder) if f.endswith('.xml')]

    for file in tqdm(files, "Sampleing files"):
        model = read_sbml_model(file)
        s = sample(model, args.n, method=args.method)
        s.to_csv(jp(args.folder, os.path.basename(file).removesuffix('.xml') + f"_{args.n}.csv"))


if __name__ == '__main__':
    main()