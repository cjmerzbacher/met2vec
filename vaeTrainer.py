
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
    def __init__(self, args):
        self.args = args

    def train(self, vae : VAE, train_fd : FluxDataset, test_fds : list[FluxDataset]):
        e_size = len(str(self.args.epochs - 1))
        with open(self.args.losses_file, "w+") as file:
            file.write("loss,reconstruction_loss,divergence_loss,test_loss\n")

        def test():
            def test_single(test_fd):
                x = test_fd.values
                y = vae.encode_decode(x)
                test_blame = vae.loss(x, y)[1]
                return test_blame
            
            test_blames = []

            for test_fd in test_fds:
                test_blame = test_single(test_fd)
                test_blames.append(test_blame)
        
        def log(i, blame, test_blames):
            test_losses = []
            for test_blame in test_blames:
                test_losses.append(test_blame['loss'])
            test_losses = np.array(test_losses)

            test_loss_mean = np.mean(test_losses)
            test_loss_max = np.max(test_losses)

            if i % self.args.save_losses_on == 0:
                with open(self.args.losses_file, "a+") as file:
                    values = [
                        blame['loss'],
                        blame['loss_reconstruction'],
                        blame['loss_divergence'],
                        test_loss_mean,
                        test_loss_max
                        ]
                    file.write(f"{','.join(map(str,values))}\n")

        def batch(i, x):
            optimizer.zero_grad()
            y = vae.encode_decode(x)
            loss, blame = vae.loss(x, y)
            loss.backward()
            optimizer.step()

            return blame


        def save_model(e):
            torch.save(vae.encoder, os.path.join(self.args.main_folder, f"encoder{e}.pth"))
            torch.save(vae.decoder, os.path.join(self.args.main_folder, f"decoder{e}.pth"))


        

        optimizer = optim.Adam(
            [
                {"params": vae.decoder.parameters()}, 
                {"params": vae.encoder.parameters()}
            ],
            lr=self.args.lr
        )

        data_loader = DataLoader(train_fd, batch_size=self.args.batch_size)

        for e in range(self.args.epochs):
            if e % self.args.refresh_data_on == 0:
                train_fd.load_sample()

            with tqdm(data_loader) as t:
                for i, (_, x) in enumerate(t):
                    blame = batch(i, x.to(device))
                    t.set_description(f"Epoch [{e:{e_size}}] loss={blame['loss']:.4e}")
            

            if e % self.args.save_on == 0:
                save_model(e)

        save_model(e)