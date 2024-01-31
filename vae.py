import torch
import torch.nn as nn
import numpy as np

from misc.constants import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def format_input(x):
    return torch.Tensor(x).to(device)

def format_matrix(A):
    return torch.Tensor(A).to(device)

def get_s(C):
    s = torch.sum(C, dim=1)
    return 1 - torch.minimum(s, torch.ones_like(s))

def format_S(S, n_outer):
    if S is None:
        return torch.zeros((n_outer, n_outer))
    return format_matrix(S)

def format_mu(v_mu, v):
    if v_mu is None:
        return torch.zeros_like(v)
    return format_matrix(v_mu)

def format_std(v_std, v):
    if v_std is None:
        return torch.zeros_like(v)
    return format_matrix(v_std)

def get_linear_network(n_in : int, n_out : int, n_lay : int, lrelu_slope : float, batch_norm : bool, dropout_p : float) -> nn.Module:
    model = []
    sizes = [round(s) for s in np.linspace(n_in, n_out, n_lay)]
    for i, (v_in, v_out) in enumerate(zip(sizes[0:-2], sizes[1:-1])):
        model += [nn.Linear(v_in, v_out), nn.LeakyReLU(negative_slope=lrelu_slope)]

        if i != 0: model += [nn.Dropout(dropout_p)]
        if batch_norm: model += [nn.BatchNorm1d(v_out)]
    model += [nn.Linear(sizes[-2], sizes[-1])]
    model = nn.Sequential(*model).float()
    model.to(device)
    return model


class FluxVAE:
    def __init__(self, 
                 n_in : int, 
                 n_emb : int, 
                 n_lay : int, 
                 lrelu_slope : float = 0.01, 
                 batch_norm : bool = False, 
                 dropout_p : float = 0.0,
                 legacy_vae : bool = False,
                 weight_decay: float = 0.5,
                 reaction_names : list[str] = None,
                 ):
        """Initializes a VAE with the dimensions and hyperparameters given."""

        self.n_in = n_in
        self.n_emb = n_emb
        self.n_lay = n_lay
        self.legacy_vae = legacy_vae
        self.lrelu_slope = lrelu_slope
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.reaction_names = reaction_names

        if reaction_names != None:
            if len(reaction_names) != n_in:
                ValueError(f"Reaction_names([{len(reaction_names)}]) isn't length of n_in:{n_in}")

        self.encoder = get_linear_network(n_in, n_emb * 2, n_lay, lrelu_slope, batch_norm, dropout_p)
        self.decoder = get_linear_network(n_emb, n_in, n_lay, lrelu_slope, batch_norm, dropout_p)

    def format_C(self, C):
        if C is None:
            return torch.eye(self.n_in)
        return format_matrix(C)


    def get_dist(self, x : torch.Tensor, C=None) -> torch.Tensor:
        """Gets the distribution values for a given input value."""
        x = format_input(x)
        C = self.format_C(C)

        y = self.encoder(x)
        mu = y[:,:self.n_emb]

        log_var = y[:,self.n_emb:]
        if  self.legacy_vae:
            log_var = torch.log(1 + y[:,self.n_emb:])

        std = torch.exp(0.5 * log_var)

        return mu, log_var, std
    
    def get_z_from_dist(self, mu, std, sample):
        ones = torch.ones_like(std)
        epsilon = torch.normal(ones, ones * 0).to(device)

        return mu + ((std * epsilon) if sample else 0.0)
    
    def encode(self, v : torch.Tensor, sample : bool = True, C=None) -> torch.Tensor:
        with torch.no_grad():
            v = format_input(v)
            C = self.format_C(C)

            v_i = torch.matmul(v, C)

            mu, _, std = self.get_dist(v_i)
            return self.get_z_from_dist(mu, std, sample)
    
    def decode(self, z : torch.Tensor, C : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            C = self.format_C(C)
            v = format_input(v)

            C_t = C.T
            v_e = v * get_s(C)

            z = format_input(z)
            v_o = self.decoder(z)

            v_r = torch.matmul(v_o, C_t) + v_e

            return v_r
    
    def train_encode_decode(self, v, C):
        """
        Returns:
            v_r: Reconstructed vector.
            mu: Mean of z distribution.
            log_var: Log of Z distribution variance.
        """
        v = format_input(v)
        C = self.format_C(C)

        C_t = torch.transpose(C, 0, 1)
        v_i = torch.matmul(v, C)
        v_e = get_s(C) * v
        
        mu, log_var, std = self.get_dist(v_i)
        z = self.get_z_from_dist(mu, std, True)

        v_o = self.decoder(z)
        v_r = torch.matmul(v_o, C_t) + v_e

        return v_r, mu, log_var 

    def loss(self, 
             v : torch.Tensor, 
             v_r : torch.Tensor, 
             mu : torch.Tensor, 
             log_var : torch.Tensor,
             S : torch.Tensor = None,
             v_mu : torch.Tensor = None,
             v_std : torch.Tensor = None,
             beta_S = 0,
             ) -> tuple[torch.Tensor, dict[str,float]]:
        """Computes the loss for a given x and y."""
        batch_size = v.shape[0]
        v = format_input(v)
        v_mu = format_mu(v_mu, v)
        v_std = format_std(v_std, v)

        S = format_S(S, v.shape[1])
        
        loss_rec = torch.sum(torch.pow(v - v_r, 2.0)) 
        loss_div = 0.5 * torch.sum(log_var.exp() + mu.pow(2) - log_var)  
        loss_S = beta_S * torch.sum(torch.pow(torch.matmul((v * v_std) + v_mu, S), 2.0))

        loss_rec /= batch_size
        loss_div /= batch_size
        loss_S /= batch_size

        loss = loss_rec + loss_div + loss_S

        def to_float(x : torch.Tensor):
            return x.detach().cpu().numpy()

        blame = {
            LOSS : to_float(loss),
            R_LOSS : to_float(loss_rec),
            D_LOSS : to_float(loss_div),
            S_LOSS : to_float(loss_S)
        }
         
        return loss, blame
    

