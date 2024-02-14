import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.vae import load_VAE, get_load_VAE_args
from misc.fluxDataset import load_fd, get_data, get_fluxes
from misc.constants import *
from misc.parsing import *
from tqdm import tqdm

from sklearn.neighbors import NearestCentroid

import numpy as np
import argparse

parser = argparse.ArgumentParser(parents=[
    PARSER_VAE_LOAD,
    PARSER_STAGE,
    parser_fluxDataset_loading("train", "-T"),
    parser_fluxDataset_loading("test", "-t"),
    PARSER_SAVE,
    PARSER_VAE_SAMPLE,
    PARSER_JOIN,
    PARSER_GROUP_BY,
])

args = parser.parse_args()

join = args.join

vae = load_VAE(*get_load_VAE_args(args))

train_fd = load_fd(args, "train", 0)
test_fd = load_fd(args, "test", 1)

fluxes = get_fluxes(train_fd, join)

sample = args.sample
stage = args.stage
group_by = args.group_by

train_df = get_data(train_fd, vae, stage, False, fluxes, source_columns=[group_by])
test_df = get_data(test_fd, vae, stage, False, fluxes, source_columns=[group_by])

train_groups = sorted(map(str, train_df[group_by].unique()))
test_groups = sorted(map(str, test_df[group_by].unique()))

train_df[group_by] = train_df[group_by].map(str).map(train_groups.index)
test_df = test_df.reindex(columns=train_df.columns)
test_df.fillna(0, inplace=True)

train_X = train_df.drop(columns=group_by).values

test_X = test_df.drop(columns=group_by).values

train_y = train_df[group_by].values


nc = NearestCentroid().fit(train_X, train_y)


test_df[f"pred_{group_by}"] = np.array(train_groups)[nc.predict(test_X)]

df = test_df.drop(columns=[c for c in test_df.columns if group_by not in c])
for i, g in enumerate(train_groups):
    df[f"{g}"] = df[f"pred_{group_by}"].map(lambda p: 1 if p == g else 0)

df = df[train_groups + [group_by]].groupby(group_by).mean(numeric_only=True).reset_index()
print(df)
df.to_csv(args.save_path, index=False)
