
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from fluxDataset import FluxDataset

from vae import VAE
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAETrainer:
    def __init__(self, args):
        self.args = args

    def train(self, vae : VAE, fd : FluxDataset):
        e_size = len(str(self.args.epochs - 1))
        with open(self.args.losses_file, "w+") as file:
            file.write("loss,reconstruction_loss,divergence_loss,test_loss\n")

        def train_batch(i, x):
            optimizer.zero_grad()
            y = vae.encode_decode(x)
            loss, blame = vae.loss(x, y)
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()

            if i % self.args.save_losses_on == 0:
                with open(self.args.losses_file, "a+") as file:
                    file.write(f"{loss},{blame['loss_reconstruction']},{blame['loss_divergence']},{test_loss}\n")

            return loss
        
        def save_model(e):
            torch.save(vae.encoder, os.path.join(self.args.main_folder, f"encoder{e}.pth"))
            torch.save(vae.decoder, os.path.join(self.args.main_folder, f"decoder{e}.pth"))

        fd.load_sample(True)
        test_set_x = torch.Tensor(fd.values)
        def test():
            x = test_set_x
            y = vae.encode_decode(x)
            test_loss = vae.loss(x, y)[0]
            return test_loss.detach().cpu().numpy()
        test_loss = test()



        optimizer = optim.Adam(
            [
                {"params": vae.decoder.parameters()}, 
                {"params": vae.encoder.parameters()}
            ],
            lr=self.args.lr
        )

        data_loader = DataLoader(fd, batch_size=self.args.batch_size)

        for e in range(self.args.epochs):
            if e % self.args.refresh_data_on == 0:
                fd.load_sample(False)

            with tqdm(data_loader) as t:
                for i, (_, x) in enumerate(t):
                    loss = train_batch(i, x.to(device))
                    t.set_description(f"Epoch [{e:{e_size}}] loss={loss:.4e}")

            test_loss = test()


            if e % self.args.save_on == 0:
                save_model(e)

        save_model(e)