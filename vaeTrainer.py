
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from fluxDataset import FluxDataset

from vae import VAE
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAETrainer:
    def __init__(self, args, vae, train_fd, test_fds):
        self.args = args
        self.vae = vae
        self.train_fd = train_fd
        self.test_fds = test_fds

        self.optimizer = optim.Adam(
            [
                {"params": self.vae.decoder.parameters()}, 
                {"params": self.vae.encoder.parameters()}
            ],
            lr=self.args.lr
        )

        self.data_loader = DataLoader(self.train_fd, batch_size=self.args.batch_size)


    def log_init(self) -> None:
        with open(self.args.losses_file, "w+") as file:
            file.write("loss,reconstruction_loss,divergence_loss,test_loss,test_loss_max\n")

    def log(self, blame, test_blames) -> None:
        test_losses = []
        for test_blame in test_blames:
            test_losses.append(test_blame['loss'])
        test_losses = np.array(test_losses)

        test_loss_mean = np.mean(test_losses)
        test_loss_max = np.max(test_losses)

        with open(self.args.losses_file, "a+") as file:
            values = [
                blame['loss'],
                blame['loss_reconstruction'],
                blame['loss_divergence'],
                test_loss_mean,
                test_loss_max
                ]
            file.write(f"{','.join(map(str,values))}\n")

    def save_model(self, e) -> None:
        torch.save(self.vae.encoder, os.path.join(self.args.main_folder, f"encoder{e}.pth"))
        torch.save(self.vae.decoder, os.path.join(self.args.main_folder, f"decoder{e}.pth"))

    def train_batch(self, x : np.array) -> dict[str,float]:
        self.optimizer.zero_grad()
        y = self.vae.encode_decode(x)
        loss, blame = self.vae.loss(x, y)
        loss.backward()
        self.optimizer.step()

        return blame

    def test_vae(self) -> list[dict[str,float]]:
        def test_single(test_fd : FluxDataset):
            x = test_fd.normalized_values
            y = self.vae.encode_decode(x)
            test_blame = self.vae.loss(x, y)[1]
            return test_blame
        
        test_blames = []

        for test_fd in self.test_fds:
            test_blame = test_single(test_fd)
            test_blames.append(test_blame)

        return test_blames

    def train(self) -> None:
        e_size = len(str(self.args.epochs - 1))
        self.log_init()

        for e in range(self.args.epochs):
            if e % self.args.refresh_data_on == 0:
                self.train_fd.load_sample()

            with tqdm(self.data_loader) as t:
                for i, (_, batch) in enumerate(t):
                    blame = self.train_batch(batch)
                    t.set_description(f"Epoch [{e:{e_size}}] loss={blame['loss']:.4e}")
                    
                    if i % self.args.save_losses_on == 0:
                        test_blames = self.test_vae()
                        self.log(blame, test_blames)
            
            if e % self.args.save_on == 0:
                self.save_model(e)

        self.save_model(e)