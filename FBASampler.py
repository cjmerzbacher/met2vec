from cobra.io import read_sbml_model
from cobra.sampling import sample, OptGPSampler, ACHRSampler
from tqdm import tqdm
from itertools import product
from misc.io import save_args

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
    parser.add_argument("--replace", action='store_true', default=False, help='Replace existing ')
    parser.add_argument("-r", "--repeats", type=int, default=1, help='Repeat sampling')
    parser.add_argument("-t", "--test", action='store_true', help="Set to print out a log of all the samples that will be made without sampling.")
    parser.add_argument("-k", type=int, default=100, help="Step size used for sampling")
    return parser.parse_args()

def get_save_models(folder, n, repeats):
    model_files = get_model_files(folder)

    save_files = [
        jp(folder, f"{os.path.basename(f)[:-4]}_{n}.csv") 
        for f in model_files
    ]

    if repeats != 1:
        model_files = model_files * repeats
        save_files = [f"{f[:-4]}({i}){f[-4:]}" for i, f in product(range(repeats), save_files)]

    return list(zip(save_files, model_files)) 

def get_model_files(folder):
    models_folder = jp(folder, MODELS_LOCATION)
    model_files = [f for f in ls(models_folder) if f.endswith('.xml')]
    model_files = [jp(models_folder, f) for f in model_files]
    return model_files

def test_print(sampled_save_models):
    for save_file, model_file in sampled_save_models:
        print(f'Sampling from model "{model_file}" -> "{save_file}"')

def get_sample(model_file, n, k, method):
    model = read_sbml_model(model_file)
    if method == 'optgp':
        sampler = OptGPSampler(model, k)
    elif method == "achr":
        sampler = ACHRSampler(model, k)

    return sampler.sample(n)

def run_sampling(args, sampled_save_models):
    with tqdm(sampled_save_models, "Sampleing files") as t:
        for save_file, model_file in t:
            save_file_bn = os.path.basename(save_file)
            model_file_bn = os.path.basename(model_file)
            t.set_description(f"{save_file_bn} <- {model_file_bn}")

            s = get_sample(model_file, args.n, args.k, args.method)
            s.to_csv(save_file)

def main():
    args = get_args()
    save_args(args.folder, args)

    save_models = get_save_models(args.folder, args.n, args.repeats)

    sampled_save_models = [
        (s, m)
        for (s, m) in save_models
        if args.replace or not os.path.exists(s)
    ]

    if args.test:
        test_print(sampled_save_models)
    else:
        run_sampling(args, sampled_save_models)



if __name__ == '__main__':
    main()