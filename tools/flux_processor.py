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
    PARSER_SAVE,
    PARSER_VAE_SAMPLE,
    PARSER_JOIN,
])
args = parser.parse_args()

join = args.join

vae = load_VAE(*get_load_VAE_args(args))

fd = load_fd(args, "", seed=0)
fluxes = get_fluxes(fd, join)

df_origin = get_data(fd, vae, args.stage, args.sample, fluxes=fluxes)
data = df_origin.values
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
df.to_csv(args.save_path, index=False)
