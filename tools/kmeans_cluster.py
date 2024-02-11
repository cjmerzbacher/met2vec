import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import pandas as pd

from sklearn.cluster import KMeans
from misc.fluxDataset  import load_fd, get_data
from misc.vae import load_VAE, get_load_VAE_args
from misc.kmeans import get_KMeans_classifications, get_k

from misc.constants import *
from misc.parsing import *

parser = argparse.ArgumentParser(parents=[
    PARSER_VAE_LOAD,
    PARSER_STAGE,
    PARSER_VAE_SAMPLE,
    parser_fluxDataset_loading(path_tag="-d"),
    PARSER_SAVE
])
parser.add_argument("-k", type=int, help="The number of clusers, defualt as many as labels in dataset.")
args = parser.parse_args()

vae = load_VAE(*get_load_VAE_args(args))
fd = load_fd(args, seed=0)

data = get_data(fd, vae, args.stage, args.sample)

k = get_k(args, fd)
kmeans_labels = get_KMeans_classifications(k, 1, data)[0]

df = pd.DataFrame({
    'label' : fd.labels, 
    KMEANS_C : kmeans_labels
})
df.to_csv(args.save_path, index=False)
