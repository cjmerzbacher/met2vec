import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import re
import pandas as pd

from vae import FluxVAE

from misc.parsing import *
from misc.vae import load_VAE
from misc.fluxDataset import load_multiple_fds

parser = argparse.ArgumentParser(parents=[
    PARSER_VAE_FOLDERS, 
    PARSER_BETA_S,
    parser_multiple_fluxDatasets_loading("test"),
    PARSER_IGNORE_CHEMICAL_NAME,
    PARSER_SAVE,
])

args = parser.parse_args()



vae_foldrs = args.vae_folders
beta_S = args.beta_S
ignore_chemical_name = args.ignore_chemical_name
save_path = args.save_path

print_args(args)

vaes = {
    folder: load_VAE(folder)
    for folder in vae_foldrs
}

fds = load_multiple_fds(args, "test", seed=0)

def remove_chemical_names(reaction_names : str):
    return [re.sub(r'n\[.*?\]', '', rn) for rn in reaction_names]

def get_vae_data(vae : FluxVAE, folder : str):
    data = vae.get_desc()
    data.pop("reaction_names")


    for fd in fds:
        if ignore_chemical_name:
            fd.core_reaction_names = remove_chemical_names(fd.core_reaction_names)
            fd.reaction_names = remove_chemical_names(fd.reaction_names)
            vae.reaction_names = remove_chemical_names(vae.reaction_names)

        fluxes = fd.reaction_names
        unfufilled_fluxes = set(vae.reaction_names).difference(fluxes)
        if len(unfufilled_fluxes) != 0:
            print(f"Warning VAE used without {len(unfufilled_fluxes)} reqired fluxes!")
            for flux in list(unfufilled_fluxes)[:1]:
                print(f"    vae - {flux}")
            for flux in list(set(fluxes).difference(vae.reaction_names))[:1]:
                pritn(f"    fd - {flux}")


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
    get_vae_data(vae, folder)
    for folder, vae in vaes.items()
])
df.to_csv(save_path, index=False)
