from vae import VAE
from fluxDataset import FluxDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import math

fd = FluxDataset("./data/samples/liver_100.csv")

dl = DataLoader(fd, batch_size=10, shuffle=True);

epochs = 5
n_in = int(fd.data.shape[1])
n_emb = 256
n_lay = 5

vae = VAE(n_in, n_emb, n_lay)

optimizer = optim.Adam(
    [
        {"params": vae.decoder.parameters()}, 
        {"params": vae.encoder.parameters()}
    ],
    lr=0.000001
)

losses = []
for epoch in range(epochs):
    with tqdm(dl) as t:
        for x in t:
            optimizer.zero_grad()

            y = vae.encode_decode(x)
            loss = vae.loss(x, y)
            loss.backward()
            losses.append(loss.detach().numpy())

            optimizer.step()

            t.set_description(f"Epoch [{epoch:3}] loss={loss:.4F}")