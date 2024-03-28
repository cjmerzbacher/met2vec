
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from fluxDataset import FluxDataset

from misc.io import safe_json_dump, safe_pkl_dump
from misc.constants import *

from vae import FluxVAE
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def divides(a, b):
    if a == 0:
        return False
    return (b % a) == 0


class VAETrainer:
    def __init__(self, 
                 vae : FluxVAE, 
                 train_fd : FluxDataset, 
                 test_fd : FluxDataset,
                 lr : float,
                 batch_size : int,
                 beta_S : float,
                 main_folder : str,
                 losses_file : str,
                 test_beta_S : float = None,
                 use_beta_S_in_test_loss : bool = False,
                 refresh_data_on : int = 0,
                 save_on : int = 0,
                 save_losses_on : int = 1,
                 save_test_min=True):
        self.set_epochs(1)

        self.lr = lr
        self.batch_size = batch_size
        self.beta_S = beta_S
        self.test_beta_S = beta_S if test_beta_S is None else test_beta_S
        self.use_beta_S_in_test_loss = use_beta_S_in_test_loss

        if use_beta_S_in_test_loss and beta_S == 0:
            print("Warning: Using beta_S in test loss while beta_S in training is 0!")

        self.refresh_data_on = refresh_data_on
        self.save_on = save_on
        self.save_losses_on = save_losses_on
        
        self.main_folder = main_folder
        self.losses_file = losses_file
        self.save_test_min = save_test_min

        self.train_fd = train_fd
        self.test_fd = test_fd

        self.vae = vae
        self.vae.set_v_mu_and_v_std(
            *self.train_fd.get_mu_std_for_reactions(self.vae.reaction_names)
        )

        self.optimizer = optim.Adam(
            [
                {"params": self.vae.decoder.parameters()}, 
                {"params": self.vae.encoder.parameters()}
            ],
            lr=self.lr,
            weight_decay=vae.weight_decay
        )

        self.data_loader = DataLoader(self.train_fd, batch_size=self.batch_size, shuffle=True)

    def set_epochs(self, epochs):
        self.epochs = epochs
        self.e_size = len(str(self.epochs - 1))

    def log_init(self) -> None:
        with open(self.losses_file, "w+") as file:
            file.write(",".join([
                "epoch",
                LOSS,
                R_LOSS,
                D_LOSS,
                S_LOSS,
                T_LOSS, 
                TR_LOSS,
                TD_LOSS,
                TS_LOSS,
                ]) + "\n")

    def log(self, epoch, blame, test_blame) -> None:
        with open(self.losses_file, "a+") as file:
            values = [
                epoch,
                blame[LOSS],
                blame[R_LOSS],
                blame[D_LOSS],
                blame[S_LOSS],
                test_blame[LOSS],
                test_blame[R_LOSS],
                test_blame[D_LOSS],
                test_blame[S_LOSS],
            ]
            file.write(f"{','.join(map(str,values))}\n")

    def save_model(self, epoch, test_min_vae=False) -> None:
        suffix = f"_testmin" if test_min_vae else f"{epoch}"

        def add_mf(path):
            return os.path.join(self.main_folder, path)

        encoder_path = add_mf(f"encoder{suffix}.pth")
        decoder_path = add_mf(f"decoder{suffix}.pth")
        vae_desc_path =add_mf(f"vae_desc{suffix}.json")
        vae_pkl_path  =add_mf(f"vae_pkl{suffix}.pkl")


        torch.save(self.vae.encoder, encoder_path)
        torch.save(self.vae.decoder, decoder_path)

        desc = self.vae.get_desc()

        desc["epoch"] = epoch

        if self.train_fd is not None:
            desc["train_dataset"] = self.train_fd.main_folder,
        if self.test_fd is not None:
            desc["test_dataset"] = self.test_fd.main_folder,

        safe_json_dump(vae_desc_path, desc, True)
        safe_pkl_dump(vae_pkl_path, self.vae, True)

    def train_batch(self, V : np.array) -> dict[str,float]:
        self.optimizer.zero_grad()
        loss, blame = self.vae.get_loss(V, self.C_train, self.S_train, self.beta_S)
        loss.backward()
        self.optimizer.step()

        return blame

    def test_vae(self) -> list[dict[str,float]]:
        if self.test_fd is None:
            return {
                LOSS : 0.0,
                R_LOSS : 0.0,
                D_LOSS : 0.0,
                S_LOSS : 0.0
            }
        with torch.no_grad():
            V = self.test_fd.values
            _, test_blame = self.vae.get_loss(V, self.C_test, self.S_test, self.test_beta_S)
            if not self.use_beta_S_in_test_loss:
                test_blame[LOSS] -= test_blame[S_LOSS]
            return test_blame
        
    def train(self, epochs : int) -> None:
        self.set_epochs(epochs)
        self.log_init()

        self.min_run_Lt = np.inf

        self.C_train = self.train_fd.get_conversion_matrix(self.vae.reaction_names)
        self.S_train = self.train_fd.S.values.T

        if self.test_fd != None:
            self.C_test = self.test_fd.get_conversion_matrix(self.vae.reaction_names)
            self.S_test = self.test_fd.S.values.T

        for e in range(self.epochs):
            self.epoch(e)
        self.save_model(e)

    def epoch(self, epoch):
        if divides(self.refresh_data_on, epoch):
            self.train_fd.reload_sample()

        min_epoch_Lt = np.inf

        with tqdm(self.data_loader) as t:
            for i, (_, X) in enumerate(t):
                desc, train_loss = self.batch(epoch, i, X)
                t.set_description(desc)

                if train_loss is not None:
                    min_epoch_Lt = min(train_loss, min_epoch_Lt)
            
        if divides(self.save_on, epoch):
            self.save_model(epoch)
        if min_epoch_Lt < self.min_run_Lt and self.save_test_min:
            self.save_test_min = min_epoch_Lt
            self.save_model(epoch, test_min_vae=True)

    def batch(self, epoch, batch, X):
        blame = self.train_batch(X)

        test_loss = None
        if divides(self.save_losses_on, batch):
            test_blame = self.test_vae()
            self.log(epoch, blame, test_blame)
            test_loss = test_blame[LOSS]

        return f"[{epoch+1:{self.e_size}}/{self.epochs}] loss={blame[LOSS]:.4e}", test_loss

                