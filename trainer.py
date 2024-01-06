import torch
import os
import argparse
import json

from vae import VAE
from vaeTrainer import VAETrainer
from torch.utils.data import DataLoader
from misc.parsing import boolean_string
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
parser.add_argument("--dataset_model_folder", help="If specified location of gem folder to be used with train dataset.")
parser.add_argument("-n", "--dataset_size", default=65536, type=int, help='The size of the dataset to be loaded for each epoch.')
parser.add_argument("--join", choices=['inner', 'outer'], default='inner', help="How the different reaction sets should be joined.")
parser.add_argument("-r", "--refresh_data_on", default=1, type=int, help="The number of epochs between changing the mix files (if used).")
parser.add_argument("--save_losses_on", type=int, default=1, help="To reduce the number of losses saved, this allows evaluations which are a multiple to be saved.")
parser.add_argument("--reload_dataset_aux", type=boolean_string, default=False, help="Used to set reload_aux on the flux dataset.")
parser.add_argument("--dataset_skip_tmp", type=boolean_string, default=False, help="If true dataset is prevented reloading tmps.")
parser.add_argument("--test_dataset_folder", help="The samples which will be used as test_sets.")
parser.add_argument("--test_dataset_model_folder", help="If specified location of gem folder to be used with test datasets.")
parser.add_argument("--test_size", type=int, default=2048, help='The size of the test sets.')
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
with open(os.path.join(args.main_folder, "args.json"), "w+") as file:
    json.dump(vars(args), file, indent=4)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}...")

# Load dataset
print("Loading dataset...")
train_fd = FluxDataset(
    args.dataset, 
    dataset_size=args.dataset_size, 
    reload_aux=args.reload_dataset_aux, 
    join=args.join, 
    verbose=True, 
    skip_tmp=args.dataset_skip_tmp,
    model_folder=args.dataset_model_folder)
n_in = train_fd.normalized_values.shape[1]

# Load test datasets
test_fds = []
test_folder = args.test_dataset_folder
if test_folder != None:
    test_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".csv")]

    for test_fd_path in test_paths:
        test_fd = FluxDataset(
            test_fd_path, 
            args.test_size, 
            join='outer', 
            columns=train_fd.columns,
            model_folder=args.test_dataset_model_folder)
        test_fds.append(test_fd)

# Load VAE
print("Loading VAE...")
vae = VAE(n_in, args.n_emb, args.n_lay, args.lrelu_slope, args.batch_norm, args.dropout)

trainer = VAETrainer(args, vae, train_fd, test_fds)
trainer.train()
