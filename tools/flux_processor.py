import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.fluxDataset import load_fd, get_data, prep_data, get_fluxes
from misc.vae import load_VAE, get_load_VAE_args
from misc.constants import *
from misc.parsing import *

import pandas as pd
import argparse

parser = argparse.ArgumentParser(parents=[
    PARSER_VAE_LOAD,
    PARSER_STAGE,
    PARSER_PREP,
    parser_fluxDataset_loading(),
    PARSER_ADD_LOSSES,
    PARSER_BETA_S,
    PARSER_SAVE,
    PARSER_VAE_SAMPLE,
    PARSER_JOIN,
    parser_seed(0),
])
args = parser.parse_args()

join = args.join
add_losses = args.add_losses
beta_S = args.beta_S
seed = args.seed

vae = load_VAE(*get_load_VAE_args(args))

fd = load_fd(args, "", seed=seed)
fluxes = get_fluxes(fd, join)

df_origin = get_data(fd, vae, args.stage, args.sample, fluxes=fluxes)
data = df_origin.values

print(f"{len(fluxes)} fluxes used for joint {join}")

data = prep_data(data, args.prep, args.perp)

if args.prep == NONE and args.stage != EMB:
     print("Using fd columns for new data.")
     columns = fluxes
else:
    prefix = args.prep
    if prefix == NONE:
        prefix = EMB
    print(f"Using prefix '{prefix}' for new data.")
    columns = [f"{prefix}{i}" for i in range(data.shape[1])]

df = pd.DataFrame(data, columns=columns)
df[SOURCE_COLUMNS] = fd.data[SOURCE_COLUMNS]

if vae is not None and add_losses:
    C = fd.get_conversion_matrix(vae.reaction_names)
    S = fd.S.values.T
    v_mu = fd.flux_mean.values
    v_std = fd.flux_std.values


    losses_df = pd.DataFrame([
        vae.get_loss(v[None,:], C, S, v_mu, v_std, beta_S)[1]
        for v in fd.normalized_values
    ])

    df = pd.concat([df, losses_df], axis=1)

df.to_csv(args.save_path, index=False)
