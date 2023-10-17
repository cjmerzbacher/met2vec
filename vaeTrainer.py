
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

    def train(self, vae : VAE, data_loader : DataLoader, epoch_update_fun = None):
        vae.to(device)
        e_size = len(str(self.args.epochs - 1))
        with open(self.args.losses_file, "w+") as file:
            file.write("loss,reconstruction_loss,divergence_loss\n")

        def train_batch(x):
            optimizer.zero_grad()
            y = vae.encode_decode(x)
            loss, reconstruction, divergence = vae.loss(x, y)
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()

            with open(self.args.losses_file, "a+") as file:
                file.write(f"{loss},{reconstruction},{divergence}\n")

            return loss
        
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

        for e in range(self.args.epochs):
            if epoch_update_fun != None and e % self.args.refresh_data_on == 0 and e != 0:
                epoch_update_fun()

            with tqdm(data_loader) as t:
                for _, x in t:
                    loss = train_batch(x.to(device))
                    t.set_description(f"Epoch [{e:{e_size}}] loss={loss:.4e}")


            if e % self.args.save_on == 0:
                save_model(e)

        save_model(e)