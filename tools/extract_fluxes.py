import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.fluxDataset import load_fd
from misc.parsing import *
from misc.constants import *

import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser(parents=[
    parser_fluxDataset_loading(),
    PARSER_FLUXES,
    parser_seed(0),
    PARSER_SAVE,
    PARSER_VERBOSE,
    parser_n("The number of fluxes sampled if none are given.")
])

args = parser.parse_args()

fluxes = args.fluxes
save_path = args.save_path
n = args.n
verbose = args.verbose

fd = load_fd(args, seed=0)

all_fluxes = fd.reaction_names

if fluxes == []:
    print("No fluxes, picking random fluxes...")
    fluxes = random.sample(all_fluxes, n)

def check_flux_verbose(flux):
    if not flux in all_fluxes:
        print(f"{flux} not found in fluxes!")
        return False
    return True

fluxes = [f for f in fluxes if check_flux_verbose(f)]

if len(fluxes) == 0:
    print("No valid fluxes, exiting.")
    quit()

print(f"Using fluxes {', '.join(fluxes)}.")

df = fd.data
df.drop(
    columns=df.columns.difference(fluxes + SOURCE_COLUMNS), 
    inplace=True
)
df = df.reindex(columns=SOURCE_COLUMNS + sorted(fluxes))

if verbose:
    print(df)

df.to_csv(save_path, index=False)
