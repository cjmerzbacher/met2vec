import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.fluxDataset import load_fd, get_data, prep_data
from misc.vae import load_VAE
from misc.constants import *
from misc.parsing import *

import pandas as pd
import argparse

parser = argparse.ArgumentParser(parents=[
    PARSER_LOAD_VAE,
    PARSER_STAGE,
    PARSER_PREP,
    fluxDataset_loading_parser(),
    PARSER_SAVE
])
args = parser.parse_args()

vae = load_VAE(args)

fd = load_fd(args, "", True)
sample = False

data = get_data(fd, vae, args.stage, sample)
data = prep_data(data, args.prep, args.perp)

if args.prep == None and args.stage != EMB:
     columns = fd.columns
else:
    pref = args.prep
    if pref == NONE:
        pref = EMB
    columns = [f"{pref}{i}" for i in range(data.shape[1])]

df = pd.DataFrame(data, columns=columns)
df["label"] = fd.labels
df.to_csv(args.save_path, index=False)
