import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from misc.parsing import *
from misc.constants import *
from misc.fluxDataset import load_fd, get_data, get_fluxes
from misc.vae import load_VAE
from misc.ari import get_bootstrap_ari
from misc.kmeans import get_KMeans_classifications, get_k

parser = argparse.ArgumentParser(parents=[
    PARSER_VAE_LOAD,
    PARSER_VAE_SAMPLE,
    parser_fluxDataset_loading(path_tag='-d'),
    PARSER_SAVE,
    PARSER_KMEANS_K,
    PARSER_JOIN,
    PARSER_STAGES,
    PARSER_ORIGIONAL_CLUSTERING,
    PARSER_BOOTSTRAP_N,
])
parser.add_argument("-n", type=int, default=64, help="Number of repititions that will be made.")
args = parser.parse_args()

vae = load_VAE(args)
fd = load_fd(args, seed=0)

k = get_k(args, fd)
n = args.n
join = args.join
bn = args.bootstrap_n
stages = args.stages
origional_clustering = args.origional_clustering

if origional_clustering not in fd.data:
    print(f"{origional_clustering} not in fd! Exiting")
    quit()

origional_types = list(fd.data[origional_clustering].unique())
fluxes = get_fluxes(fd, join)

dfs = {
    stage : get_data(fd, vae, stage, args.sample, fluxes=fluxes)
    for stage in stages
}

clustering_sets = {
    stage : get_KMeans_classifications(k, n, dfs[stage])
    for stage in stages
}
clustering_sets[ORIGIONAL] = [[origional_types.index(cell) for cell in fd.data[origional_clustering]]]

labels = clustering_sets.keys()
n_labels = len(labels)

data = np.zeros((n_labels * 2, n_labels))
columns = sum([[label, label+ '_std'] for label in labels], start=[])

rows = labels 

for i, a in tqdm(list(enumerate(labels)), desc="Calculating ARIs"):
    for j, b in enumerate(labels): 
        mean, std = get_bootstrap_ari(
            clustering_sets[a], 
            clustering_sets[b], 
            bn
        )

        data[i*2, j] = mean
        data[i*2 + 1, j] = std

df = pd.DataFrame(data.T, columns=columns)
df['stage'] = rows
df.to_csv(args.save_path, index=False)
