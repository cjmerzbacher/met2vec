import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.gmms import train_gmms
from misc.vae import load_VAE
from misc.fluxDataset import load_fd, get_data
from misc.constants import *
from misc.parsing import *
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
    ])

args = parser.parse_args()

vae = load_VAE(args)

train_fd = load_fd(args, "train", seed=0)
train_columns = train_fd.columns

test_fd = load_fd(args, "test", seed=1)
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