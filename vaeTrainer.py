
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from fluxDataset import FluxDataset

from vae import VAE
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def divides(a, b):
    if a == 0:
        return False
    return b % a == 0


class VAETrainer:
    def __init__(self, args, vae, train_fd, test_fd):
        self.args = args

        self.epochs = args.epochs
        self.e_size = len(str(self.epochs - 1))

        self.vae : VAE = vae
        self.train_fd = train_fd
        self.test_fd = test_fd

        self.optimizer = optim.Adam(
            [
                {"params": self.vae.decoder.parameters()}, 
                {"params": self.vae.encoder.parameters()}
            ],
            lr=self.args.lr
        )

        self.data_loader = DataLoader(self.train_fd, batch_size=self.args.batch_size, shuffle=True)


    def log_init(self) -> None:
        with open(self.args.losses_file, "w+") as file:
            file.write(",".join([
                "epoch",
                "loss",
                "reconstruction_loss",
                "divergence_loss",
                "test_loss",
                "test_rec",
                "test_div\n",
                ]))

    def log(self, epoch, blame, test_blame) -> None:
        with open(self.args.losses_file, "a+") as file:
            values = [
                epoch,
                blame['loss'],
                blame['loss_reconstruction'],
                blame['loss_divergence'],
                test_blame['loss'],
                test_blame['loss_reconstruction'],
                test_blame['loss_divergence'],
                ]
            file.write(f"{','.join(map(str,values))}\n")

    def save_model(self, epoch) -> None:
        torch.save(self.vae.encoder, os.path.join(self.args.main_folder, f"encoder{epoch}.pth"))
        torch.save(self.vae.decoder, os.path.join(self.args.main_folder, f"decoder{epoch}.pth"))



    def train_batch(self, x : np.array) -> dict[str,float]:
        self.optimizer.zero_grad()
        y = self.vae.encode_decode(x)
        loss, blame = self.vae.loss(x, y)
        loss.backward()
        self.optimizer.step()

        return blame

    def test_vae(self) -> list[dict[str,float]]:
        x = self.test_fd.normalized_values
        y = self.vae.encode_decode(x)
        test_blame = self.vae.loss(x, y)[1]
        return test_blame
        
    def train(self) -> None:
        self.log_init()

        for e in range(self.epochs):
            self.epoch(e)
        self.save_model(e)

    def epoch(self, epoch):
        
        if divides(self.args.refresh_data_on, epoch):
            self.train_fd.load_sample()

        with tqdm(self.data_loader) as t:
            for i, (_, X) in enumerate(t):
                desc = self.batch(epoch, i, X)
                t.set_description(desc)
            
        if divides(self.args.save_on, epoch):
            self.save_model(epoch)

    def batch(self, epoch, batch, X):
        blame = self.train_batch(X)

        if divides(self.args.save_losses_on, batch):
            test_blame = self.test_vae()
            self.log(epoch, blame, test_blame)

        return f"[{epoch+1:{self.e_size}}/{self.epochs}] loss={blame['loss']:.4e}"
                