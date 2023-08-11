import torch
import os
import argparse

from vae import VAE
from vaeTrainer import VAETrainer
from torch.utils.data import DataLoader
from fluxDataset import FluxDataset
from datetime import datetime
from collections import namedtuple

# Get arguments
parser = argparse.ArgumentParser("VAE trainer", "Python program to train VAE from flux dataset.")
parser.add_argument("-e", "--epochs", default=1, type=int)
parser.add_argument("-s", "--save_on", default=10, type=int)
parser.add_argument("--n_emb", default=128, type=int)
parser.add_argument("--n_lay", default=5, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("-d" ,"--dataset_name", required=True, type=str)
args = parser.parse_args()

# Setup model folder
args.model_folder = os.path.abspath(os.path.join(
        "data", 
        "models", 
        args.dataset_name, 
        datetime.now().strftime('%m-%d-%Y@%H-%M-%S')),
    )
os.makedirs(args.model_folder)

# Setup losses file
args.losses_file = os.path.join(args.model_folder, "losses.csv")

# Save args
with open(os.path.join(args.model_folder, "args.txt"), "w+") as file:
    file.writelines([f"{n}:{v}\n" for n, v, in args._get_kwargs()])

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}...")

# Load dataset
print("Loading dataset...")
fd = FluxDataset(f"./data/samples/{args.dataset_name}.csv")
dl = DataLoader(fd, batch_size=1, shuffle=True);
n_in = int(fd.data.shape[1])

# Load VAE
print("Loading VAE...")
vae = VAE(n_in, args.n_emb, args.n_lay)

trainer = VAETrainer(args)
trainer.train(vae, dl)
