import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vae import VAE
from fluxDataset import FluxDataset
from tqdm import tqdm

import os
import argparse

# Get arguments
parser = argparse.ArgumentParser("VAE trainer", "Python program to train VAE from flux dataset.")
parser.add_argument("-e", "--epochs", default=1, type=int)
parser.add_argument("-s", "--save_on", default=10, type=int)
parser.add_argument("--n_emb", default=128, type=int)
parser.add_argument("--n_lay", default=5, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("-d" ,"--dataset_name", required=True, type=str)
args = parser.parse_args()

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Setup model folder
model_folder = os.path.join("data/models", args.dataset_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Load dataset
print("Loading dataset...")
fd = FluxDataset(f"./data/samples/{args.dataset_name}.csv")
dl = DataLoader(fd, batch_size=64, shuffle=True);
n_in = int(fd.data.shape[1])

vae = VAE(n_in, args.n_emb, args.n_lay)
vae.encoder = vae.encoder.to(device)
vae.decoder = vae.decoder.to(device)

optimizer = optim.Adam(
    [
        {"params": vae.decoder.parameters()}, 
        {"params": vae.encoder.parameters()}
    ],
    lr=args.lr
)

losses = []
for epoch in range(args.epochs):
    with tqdm(dl) as t:
        for x in t:
            x = x.to(device)
            optimizer.zero_grad()

            y = vae.encode_decode(x)
            loss = vae.loss(x, y)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())

            optimizer.step()

            t.set_description(f"Epoch [{epoch:3}] loss={loss:.4e}")

    if epoch % args.save_on == 0:
        torch.save(vae.encoder, os.path.join(model_folder, f"encoder{epoch}.pth"))
        torch.save(vae.decoder, os.path.join(model_folder, f"decoder{epoch}.pth"))

with open(os.path.join(model_folder, "losses.csv"), 'w+') as file:
    file.write("loss\n")
    file.writelines([f"{l}\n" for l in losses])
