
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vae import VAE
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAETrainer:
    def __init__(self, args):
        self.args = args

    def train(self, vae : VAE, data_loader : DataLoader):
        e_size = len(str(self.args.epochs - 1))
        with open(self.args.losses_file, "w+") as file:
            file.write("loss\n")

        def train_batch(x):
            optimizer.zero_grad()
            loss = vae.forward_backward(x)
            optimizer.step()

            with open(self.args.losses_file, "a+") as file:
                file.write(f"{loss}\n")

            return loss
        
        def save_model(e):
            torch.save(vae.encoder, os.path.join(self.args.main_folder, f"encoder{e}.pth"))
            torch.save(vae.decoder, os.path.join(self.args.main_folder, f"decoder{e}.pth"))

        optimizer = optim.SGD(
            [
                {"params": vae.decoder.parameters()}, 
                {"params": vae.encoder.parameters()}
            ],
            lr=self.args.lr
        )

        for e in range(self.args.epochs):
            with tqdm(data_loader) as t:
                for x in t:
                    loss = train_batch(x.to(device))
                    t.set_description(f"Epoch [{e:{e_size}}] loss={loss:.4e}")

            if e % self.args.save_on:
                save_model(e)

        save_model(e)