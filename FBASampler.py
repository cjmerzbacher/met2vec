from cobra.io import read_sbml_model
from cobra.sampling import sample
from tqdm import tqdm
from itertools import product

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
    parser.add_argument("-r", "--replace", action='store_true', default=False, help='Replace existing ')
    parser.add_argument("-t", "--test_repeats", type=int, default=1, help='Repeat sampling')

    parser.add_argument("-p", "--print_samples", action='store_true', help="Set to print out a log of all the samples that will be made without sampling.")
    return parser.parse_args()

def main():
    args = get_args()

    models_folder = jp(args.folder, MODELS_LOCATION)
    file_names = [f for f in ls(models_folder) if f.endswith('.xml')]
    model_files = [jp(models_folder, f) for f in file_names]

    save_files = [jp(args.folder, f"{f[:-4]}_{args.n}.csv") for f in file_names]
    if args.test_repeats != 1:
        model_files = model_files * args.test_repeats
        save_files = [f"{f[:-4]}({i}){f[-4:]}" for f, i in product(save_files, range(args.test_repeats))]

    file_indices = [i for i in range(len(save_files)) if args.replace or not os.path.exists(save_files[i])]

    for i in tqdm(file_indices, "Sampleing files", disable=args.print_samples):
        if args.print_samples:
            print(f'Sampling from model "{model_files[i]}" -> "{save_files[i]}"')
            continue

        model = read_sbml_model(model_files[i])
        s = sample(model, args.n, method=args.method)
        s.to_csv(save_files[i])

if __name__ == '__main__':
    main()