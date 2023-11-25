import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.gmms import train_gmms
from misc.vae import PARSER_LOAD_VAE, load_VAE
from misc.constants import *
from misc.fluxDataset import load_fd, get_data
from misc.parsing import boolean_string, fluxDataset_loading_parser, get_save_parser
from tqdm import tqdm

import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(parents=[
    PARSER_LOAD_VAE,
    fluxDataset_loading_parser("train", "-T"),
    fluxDataset_loading_parser("test", "-t"),
    get_save_parser(),
    ])

parser.add_argument("-s", "--stage", choices=VAE_STAGES, help="The stage in the VAE used to evaluated data.")
parser.add_argument("--plot_values", type=boolean_string, default=False, help="Whether the values should be plotted")
args = parser.parse_args()

vae = load_VAE(args)

train_fd = load_fd(args, "train", True)
train_columns = train_fd.columns

test_fd = load_fd(args, "test", True)
test_fd.set_columns(train_columns)
sample = False

gmms = train_gmms(train_fd, vae, args.stage, sample)

test_labels =   test_fd.unique_labels
train_labels = train_fd.unique_labels

nt = len(test_labels)
nT = len(train_labels)

test_data_sets = {label : get_data(test_fd, vae, args.stage, sample, label) for label in test_labels}

def pred(data):
    probs = np.array([
            gmms[label].score_samples(data).ravel() 
        for label in train_labels])

    return np.argmax(probs, axis=0)

def get_prediction_accuracy(exp_label, data_label):
    return np.mean(pred(test_data_sets[data_label]) == train_labels.index(exp_label))

accuracies = np.zeros((nt, nT))
for i, test_label in tqdm(list(enumerate(test_labels)), desc="Calculating Acc"):
    for j, train_label in enumerate(train_labels):
        accuracies[i,j] = get_prediction_accuracy(train_label, test_label)

df = pd.DataFrame(data=accuracies, columns=train_labels)
df['test_label'] = test_labels
df.to_csv(args.save_path, index=False)