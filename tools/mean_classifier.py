import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.vae import load_VAE
from misc.fluxDataset import load_fd, get_data, get_fluxes
from misc.constants import *
from misc.parsing import *
from misc.classifier import *
from tqdm import tqdm

import numpy as np
import pandas as pd
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

vae = load_VAE(args)

train_fd = load_fd(args, "train", 0)
test_fd = load_fd(args, "test", 1)

fluxes = get_fluxes(train_fd, join)

sample = args.sample
stage = args.stage
group_by = args.group_by

train_df = get_data(train_fd, vae, stage, False, fluxes)
test_df = get_data(test_fd, vae, stage, False, fluxes)

pred = get_mean_pred(train_df, group_by)

df = get_prediction_df(test_df, train_df, group_by, pred)
df.to_csv(args.save_path, index=False)
