import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.constants import *
from misc.vae import read_VAE_args
from misc.parsing import *

import argparse
import pandas as pd

joinp = os.path.join

parser = argparse.ArgumentParser(parents=[
    PARSER_SAVE
])

parser.add_argument("folders", nargs='+', help='The folders trained VAEs should be in.')
parser.add_argument("-a", "--average_over", type=int, default=128, help="The number of loss evaluations that will be averaged over.")
args = parser.parse_args()

folders = args.folders
loss_paths = [joinp(f, LOSSES_PATH) for f in folders]
args_paths = [joinp(f, ARGS_PATH) for f in folders]

avg_over = args.average_over

loss_ends = []
for lp, ap in zip(loss_paths, args_paths):
    loss = pd.read_csv(lp)
    rargs = read_VAE_args(ap)

    n_loss = len(loss)
    if n_loss < avg_over:
        print(f"With '{lp}' only {n_loss} evaluations {avg_over} required.")
        quit()

    s_loss_end = loss[loss.index > n_loss - avg_over].mean(axis=0)
    for name, value in rargs.items():
        s_loss_end[name] = value

    loss_ends.append(s_loss_end)

df = pd.DataFrame(loss_ends)
df.to_csv(args.save_path, index=False)
