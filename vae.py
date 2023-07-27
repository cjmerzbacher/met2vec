import torch
import torch.nn as nn
import numpy as np

class VAE:
    def __init__(self, n_in, n_emb, n_lay):
        self.n_in = n_in
        self.n_emb = n_emb
        self.n_lay = n_lay

        encoder_sizes = [round(i) for i in np.linspace(n_in, n_emb * 2, n_lay)]
        encoder = []
        for v_in, v_out in zip(encoder_sizes[0:-1], encoder_sizes[1:]):
            encoder += [
                nn.Linear(v_in, v_out),
                nn.ReLU()
            ]
        self.encoder = nn.Sequential(*encoder)


        decoder_sizes = [round(i) for i in np.linspace(n_in, n_emb, n_lay)]
        decoder = []
        for v_in, v_out in zip(decoder_sizes[-1:0:-1], decoder_sizes[-2::-1]):
            decoder += [
                nn.Linear(v_in, v_out),
                nn.ReLU()
            ]
        self.decoder = nn.Sequential(*decoder)

    def get_dist(self, x):
        y = self.encoder(x)
        self.mu = y[:self.n_emb]
        self.sigma = torch.pow(y[self.n_emb:], 0.5)

        return self.mu, self.sigma

    def encoder_decode(self, x):
        mu, sigma = self.get_dist(x) 
        epsilon = torch.normal(torch.zeros(self.n_emb), torch.ones(self.n_emb)) 

        z = mu + sigma * epsilon

        y = self.decoder(z)
        return y
    
    def loss(self, x, y):
        # Reconstruction loss
        loss = torch.pow(x - y, 2.0) 

        # Divergence from N(0, 1)
        loss += 0.5 * torch.sum(self.sigma) # tr(sigma)
        loss += 0.5 * torch.dot(self.mu, self.mu) # mu @ mu'
        loss += 0.5 * torch.sum(torch.log(self.sigma))

        return loss
