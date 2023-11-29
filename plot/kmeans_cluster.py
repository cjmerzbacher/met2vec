import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import pandas as pd

from sklearn.cluster import KMeans
from misc.fluxDataset  import load_fd, get_data
from misc.vae import load_VAE

from misc.constants import *
from misc.parsing import *

parser = argparse.ArgumentParser(parents=[
    PARSER_LOAD_VAE,
    PARSER_STAGE,
    PARSER_SAMPLE,
    fluxDataset_loading_parser(path_tag="-d"),
    PARSER_SAVE
])
parser.add_argument("-k", type=int, help="The number of clusers, defualt as many as labels in dataset.")
args = parser.parse_args()

vae = load_VAE(args)
fd = load_fd(args)

data = get_data(fd, vae, args.stage, args.sample)

k = args.k if args.k != None else len(fd.unique_labels)
kmeans = KMeans(k, n_init='auto').fit(data)

df = pd.DataFrame({
    'label' : fd.labels, 
    KMEANS_C : kmeans.labels_
})

df[KMEANS_C] = kmeans.labels_
df.to_csv(args.save_path, index=False)
