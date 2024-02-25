import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import pandas as pd

from vae import FluxVAE

from misc.parsing import *
from misc.vae import load_VAE
from misc.fluxDataset import load_multiple_fds

parser = argparse.ArgumentParser(parents=[
    PARSER_VAE_FOLDERS, 
    PARSER_BETA_S,
    parser_multiple_fluxDatasets_loading("test"),
    PARSER_SAVE,
])

args = parser.parse_args()



vae_foldrs = args.vae_folders
beta_S = args.beta_S
save_path = args.save_path

print_args(args)

vaes = [
    load_VAE(folder)
    for folder in vae_foldrs
]

fds = load_multiple_fds(args, "test", seed=0)

def get_vae_data(vae : FluxVAE):
    data = vae.get_desc()
    data.pop("reaction_names")

    for fd in fds:
        _, blame = vae.get_loss(
            fd.normalized_values, 
            fd.get_conversion_matrix(vae.reaction_names),
            fd.S.values.T,
            fd.flux_mean.values,
            fd.flux_std.values,
            beta_S)
        data[fd.main_folder] = blame[LOSS]

    return data

df = pd.DataFrame([
    get_vae_data(vae)
    for vae in vaes
])
df.to_csv(save_path, index=False)
