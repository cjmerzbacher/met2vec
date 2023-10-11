import torch
import torch.nn as nn
import numpy as np
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_VAE_from_args(n_in : int, path : str):
    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except ValueError:    
            date = {}
            for line in file.readlines():
                name, value = line.split(':')
                match name:
                    case 'n_emb':
                        value = int(value)
                    case 'n_lay':
                        value = int(value)
                    case 'lrelu_slope':
                        value = float(value)
                    case _:
                        value = None
                if value: 
                    data[name] = value
        return VAE(n_in, data['n_emb'], data['n_lay'], data['lrelu_slope'])
        


class VAE:
    def __init__(self, n_in : int, n_emb : int, n_lay : int, lrelu_slope : float):
        self.n_in = n_in
        self.n_emb = n_emb
        self.n_lay = n_lay

        encoder_sizes = [round(i) for i in np.linspace(n_in, n_emb * 2, n_lay)]
        encoder = []
        for v_in, v_out in zip(encoder_sizes[0:-2], encoder_sizes[1:-1]):
            encoder += [
                nn.Linear(v_in, v_out),
                nn.ReLU()
            ]
        encoder += [
            nn.Linear(encoder_sizes[-2], encoder_sizes[-1])
        ]
        self.encoder = nn.Sequential(*encoder).float()


        decoder_sizes = [round(i) for i in np.linspace(n_in, n_emb, n_lay)]
        decoder = []
        for v_in, v_out in zip(decoder_sizes[-1:1:-1], decoder_sizes[-2:0:-1]):
            decoder += [
                nn.Linear(v_in, v_out),
                nn.LeakyReLU(negative_slope=lrelu_slope)
            ]
        decoder += [
            nn.Linear(decoder_sizes[1], decoder_sizes[0])
        ]
        self.decoder = nn.Sequential(*decoder).float()

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def get_dist(self, x):
        y = self.encoder(x)
        self.mu = y[:,:self.n_emb]
        self.sigma = torch.log(1.0 + torch.exp(y[:,self.n_emb:])) #Soft plus

        return self.mu, self.sigma

    def encode_decode(self, x):
        global device

        mu, sigma = self.get_dist(x) 
        epsilon = torch.normal(torch.zeros(sigma.size()), torch.ones(sigma.size())).to(device)

        z = mu + (sigma * epsilon)

        y = self.decoder(z)
        return y

    def loss(self, x, y):
        # Reconstruction loss
        loss_reconstruction = torch.sum(torch.pow(x - y, 2.0), dim=1) 
        loss_reconstruction = torch.mean(loss_reconstruction)

        # Divergence from N(0, 1)
        loss_divergence = 0.5 * torch.sum(self.sigma, dim=1) # tr(sigma)
        loss_divergence += 0.5 * torch.norm(self.mu, dim=1)
        loss_divergence -= 0.5 * torch.sum(torch.log(self.sigma + 0.0001), dim=1) #log(sigma)
        loss_divergence = torch.mean(loss_divergence)


        loss = loss_reconstruction + loss_divergence
         
        return loss, loss_reconstruction.detach().cpu().numpy(), loss_divergence.detach().cpu().numpy()