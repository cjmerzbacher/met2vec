import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import numpy as np
import pandas as pd

from misc.parsing import *
from misc.fluxDataset import *
from misc.vae import *
from misc.kmeans import *

from tqdm import tqdm

parser = argparse.ArgumentParser(parents=[
    parser_fluxDataset_loading(path_tag="-d"),
    PARSER_VAE_LOAD,
    PARSER_VAE_SAMPLE,
    PARSER_MAX_K,
    PARSER_STAGES,
    PARSER_JOIN,
    PARSER_SAVE,
    parser_n("The number of k means clustering that will be performed to get std.", default=8)
])
args = parser.parse_args()


join = args.join
stages = args.stages
sample = args.sample
n = args.n
save_path = args.save_path

fd = load_fd(args, seed=0)
fluxes = get_fluxes(fd, join)
vae = load_VAE(args)
max_k = get_max_k(args, fd)

dfs = get_data_at_stages(fd, vae, stages, sample)

ks = list(range(1, max_k+1))

n_stages = len(stages)
n_ks = len(ks)

data = np.zeros((n_ks, n_stages))
for i, k in tqdm(list(enumerate(ks)), desc='Runing KMeans'):
    for j, stage in enumerate(stages):
        data[i,j] = get_kmeans_inertia(k, dfs[stage]) / len(dfs[stage].index)

elbo_df = pd.DataFrame(
    data,
    columns=map(str,stages)
)
elbo_df['k'] = ks
elbo_df.to_csv(save_path, index=False)

