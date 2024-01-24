import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.constants import *
from misc.vae import read_VAE_args
from misc.parsing import *

import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm

joinp = os.path.join

parser = argparse.ArgumentParser(parents=[
    PARSER_SAVE
])

parser.add_argument("folders", nargs='+', help='The folders trained VAEs should be in.')
parser.add_argument("-a", "--average_over", type=int, default=128, help="The number of loss evaluations that will be averaged over.")
parser.add_argument("-e", "--average_over_epoch", action="store_true", default=False, help="If set values will be averaged over epochs.")
args = parser.parse_args()

folders = args.folders

for fo in folders:
    if fo.endswith('*'):
        f = fo.removesuffix('*')
        subfolders = filter(os.path.isdir, [os.path.join(f, sf) for sf in os.listdir(f)])
        folders.remove(fo)
        folders += subfolders

loss_paths = [joinp(f, LOSSES_PATH) for f in folders]
args_paths = [joinp(f, ARGS_PATH) for f in folders]

avg_over = args.average_over
avg_over_epoch = args.average_over_epoch

loss_ends = []
it = enumerate(
    list(zip(loss_paths, args_paths)), 
)

for i, (lp, ap) in tqdm(it, desc='Processing losses.csv file(s)...'):
    try:
        loss = pd.read_csv(lp)
        rargs = read_VAE_args(ap)
    except:
        print(f"Unable to load run of '{lp}', '{ap}'.")
        continue

    n_loss = len(loss)
    if avg_over_epoch and "epoch" in loss.columns:
        s_loss_end = loss.groupby("epoch").agg(np.mean).reset_index()
    elif n_loss >= avg_over:
        s_loss_end = loss[loss.index > n_loss - avg_over].mean(axis=0)
    else:
        print(f"With '{lp}' only {n_loss} evaluations {avg_over} required.")
        quit()

    for name, value in rargs.items():
        s_loss_end[name] = value
    s_loss_end['run'] = i

    loss_ends.append(s_loss_end)

df = pd.concat(loss_ends, join='outer')
df.to_csv(args.save_path, index=False)
