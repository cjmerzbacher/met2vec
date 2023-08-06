import torch
from vae import VAE
from fluxDataset import FluxDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device {device}")

fd = FluxDataset("./data/samples/liver_10k.csv")
dl = DataLoader(fd, batch_size=64, shuffle=True);

epochs = 50
n_in = int(fd.data.shape[1])
n_emb = 256
n_lay = 5

vae = VAE(n_in, n_emb, n_lay)
vae.encoder = vae.encoder.to(device)
vae.decoder = vae.decoder.to(device)

optimizer = optim.Adam(
    [
        {"params": vae.decoder.parameters()}, 
        {"params": vae.encoder.parameters()}
    ],
    lr=0.0001
)

losses = []
for epoch in range(epochs):
    with tqdm(dl) as t:
        for x in t:
            x = x.to(device)
            optimizer.zero_grad()

            y = vae.encode_decode(x)
            loss = vae.loss(x, y)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())

            optimizer.step()

            t.set_description(f"Epoch [{epoch:3}] loss={loss:.4F}")