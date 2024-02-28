import torch
import os
import argparse
import json

from vae import FluxVAE
from vaeTrainer import VAETrainer
from torch.utils.data import DataLoader
from misc.parsing import boolean_string
from misc.io import save_args
from misc.constants import *
from fluxDataset import FluxDataset

# Get arguments
parser = argparse.ArgumentParser("VAE trainer", "Python program to train VAE from flux dataset.")
parser.add_argument("-e", "--epochs", default=1, type=int, help="The number of epochs the VAE will be trained for.")
parser.add_argument("-b", "--batch_size", default=64, type=int, help="The batch size samples will be taken in.")
parser.add_argument("-s", "--save_on", default=10, type=int, help="The number of epochs between saves of the VAE.")
parser.add_argument("--n_emb", default=128, type=int, help="The number of embeding dimensions.")
parser.add_argument("--n_lay", default=5, type=int, help="The number of layers.")
parser.add_argument("--lr", default=0.0001, type=float, help="The step size / learning rate used in SGD.")
parser.add_argument("--lrelu_slope", type=float, default=0.0, help="The lrelu-slop used in the VAE.")
parser.add_argument("--batch_norm", type=boolean_string, default=False, help="Wether or not batch norm will be applied imbetween layers")
parser.add_argument("--dropout", default=0.0, type=float, help="The probability of a node being dropped out while training")
parser.add_argument("-d" ,"--dataset", required=True, type=str, help="The directory or file the dataset is saved in.")
parser.add_argument("--model_folder", help="If specified location of gem folder to be used with train dataset.")
parser.add_argument("-n", "--dataset_size", default=65536, type=int, help='The size of the dataset to be loaded for each epoch.')
parser.add_argument("--join", choices=['inner', 'outer'], default='inner', help="How the different reaction sets should be joined.")
parser.add_argument("-r", "--refresh_data_on", default=1, type=int, help="The number of epochs between changing the mix files (if used).")
parser.add_argument("--save_losses_on", type=int, default=1, help="To reduce the number of losses saved, this allows evaluations which are a multiple to be saved.")
parser.add_argument("--test_dataset", help="The samples which will be used as test_sets.")
parser.add_argument("--test_size", type=int, default=2048, help='The size of the test sets.')
parser.add_argument("--save_test_min", type=boolean_string, default=True, help="If true will save the vae which scored the lowest loss on test data (default True)")#
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay value.")
parser.add_argument("--beta_S", type=float, default=0.0, help="Weighting value for the stoicheometry loss.")
parser.add_argument("--test_beta_S", type=float, default=0.0, help="Beta_S value output in test_loss")
parser.add_argument("main_folder", type=str, help="Name of the folder data will be saved to.")
args = parser.parse_args()

# Setup model folder
if args.main_folder is None:
    args.main_folder = "%m/%d/%Y, %H:%M:%S"

if not os.path.exists(args.main_folder):
    os.makedirs(args.main_folder)

# Setup losses file
args.losses_file = os.path.join(args.main_folder, "losses.csv")

# Save args
save_args(args.main_folder, args)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}...")

# Load dataset
print("Loading dataset...")
train_fd = FluxDataset(
    args.dataset, 
    n=args.dataset_size, 
    model_folder=args.model_folder
)

# Find Reactions VAE will learn to reconstruct
vae_reactions = train_fd.core_reaction_names if args.join == INNER else train_fd.reaction_names
n_in =len(vae_reactions)

print(f"    {len(train_fd.reaction_names)} total reactions")
print(f"    {n_in} VAE reactions")

# Load test datasets
if args.test_dataset != None:
    test_fd = FluxDataset(
        args.test_dataset, 
        args.test_size, 
        model_folder=args.model_folder
    )
else:
    test_fd = None

# Load VAE
print("Loading VAE...")
vae = FluxVAE(
    n_in=n_in, 
    n_emb=args.n_emb, 
    n_lay=args.n_lay, 
    lrelu_slope=args.lrelu_slope, 
    batch_norm=args.batch_norm, 
    dropout_p=args.dropout, 
    weight_decay=args.weight_decay,
    reaction_names=vae_reactions
)

trainer = VAETrainer(
    vae=vae, 
    train_fd=train_fd, 
    test_fd=test_fd,
    lr=args.lr,
    batch_size=args.batch_size,
    beta_S=args.beta_S,
    main_folder=args.main_folder,
    losses_file=args.losses_file,
    test_beta_S=args.test_beta_S,
    refresh_data_on=args.refresh_data_on,
    save_on=args.save_on,
    save_losses_on=args.save_losses_on,
    save_test_min=args.save_test_min
)
trainer.train(args.epochs)
