import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from itertools import product

from misc.parsing import *
from misc.constants import *
from misc.fluxDataset import load_fd, get_data
from misc.vae import load_VAE
from misc.ari import get_bootstrap_ari
from misc.kmeans import get_KMeans_classifications, get_k

parser = argparse.ArgumentParser(parents=[
    PARSER_LOAD_VAE,
    PARSER_SAMPLE,
    fluxDataset_loading_parser(path_tag='-d'),
    PARSER_SAVE,
    PARSER_KMEANS_K,
])
parser.add_argument("-n", type=int, default=64, help="Number of repititions that will be made.")
parser.add_argument("--bootstrap_n", type=int, default=128, help="Number of bootstrap repititions that will be made to calculate mean and variance for ari.")
args = parser.parse_args()

vae = load_VAE(args)
fd = load_fd(args, seed=0)

k = get_k(args, fd)
n = args.n
bn = args.bootstrap_n
cell_types = fd.unique_labels

data_pre = get_data(fd, vae, PRE, args.sample)
data_emb = get_data(fd, vae, EMB, args.sample)
data_rec = get_data(fd, vae, REC, args.sample)

sets = {
    PRE : get_KMeans_classifications(k, n, data_pre),
    EMB : get_KMeans_classifications(k, n, data_emb),
    REC : get_KMeans_classifications(k, n, data_rec),
    ORIGIONAL : [[cell_types.index(cell) for cell in fd.labels]],
}

set_labels = sets.keys()
n_lab = len(set_labels)

data = np.zeros((n_lab * 2, n_lab))
data_columns = sum([[l, l + '_std'] for l in set_labels], start=[])
data_rows = set_labels

for i, a in tqdm(enumerate(set_labels), desc="Calculating ARIs"):
    for j, b in enumerate(set_labels): 
        mean, std = get_bootstrap_ari(sets[a], sets[b], bn)
        data[i*2, j] = mean
        data[i*2 + 1, j] = std

df = pd.DataFrame(data.T, columns=data_columns)
df['stage'] = data_rows
df.to_csv(args.save_path, index=False)
