import torch
import torch.nn as nn
import numpy as np

class VAE:
    def __init__(self, n_in : int, n_emb : int, n_lay : int):
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
                nn.ReLU()
            ]
        decoder += [
            nn.Linear(decoder_sizes[1], decoder_sizes[0])
        ]
        self.decoder = nn.Sequential(*decoder).float()

    def get_dist(self, x):
        y = self.encoder(x)
        self.mu = y[:,:self.n_emb]
        self.sigma = torch.pow(torch.sigmoid(y[:,self.n_emb:]), 0.5)

        return self.mu, self.sigma

    def encode_decode(self, x):
        mu, sigma = self.get_dist(x) 
        epsilon = torch.normal(torch.zeros(sigma.size()), torch.ones(sigma.size())) 

        z = mu + (sigma * epsilon)

        y = self.decoder(z)
        return y

    def loss(self, x, y):
        # Reconstruction loss
        loss = torch.sum(torch.pow(x - y, 2.0)) 

        # Divergence from N(0, 1)
        loss += 0.5 * torch.sum(self.sigma) # tr(sigma)
        loss += 0.5 * torch.sum(torch.norm(self.mu, dim=1))
        loss -= 0.5 * torch.sum(torch.log(self.sigma + 0.0001)) #log(sigma)

        return loss
