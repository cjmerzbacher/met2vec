import torch
import os
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAE:
    def __init__(self, n_in : int, n_emb : int, n_lay : int, lrelu_slope : float = 0.01):
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
        self.encoder : nn.Module = nn.Sequential(*encoder).float()


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
        self.decoder : nn.Module = nn.Sequential(*decoder).float()

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def get_dist(self, x):
        y = self.encoder(torch.Tensor(x))
        self.mu = y[:,:self.n_emb]
        self.sigma = torch.log(1.0 + torch.exp(y[:,self.n_emb:])) #Soft plus

        return self.mu, self.sigma
    
    def encode(self, x, sample=True):
        global device

        mu, sigma = self.get_dist(x) 
        epsilon = torch.normal(torch.zeros(sigma.size()), torch.ones(sigma.size())).to(device)

        return mu + ((sigma * epsilon) if sample else 0.0)


    def encode_decode(self, x):
        global device

        z = self.encode(x)
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
    

def make_VAE(folder : str, load_version : int = None) -> VAE:
    def contains_correct_version(f : str, model_part : str):
        return model_part in f and (True if load_version is None else str(load_version) in f)
    
    encoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'encoder')]
    decoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'decoder')]

    if len(encoder_files) == 0 or len(decoder_files) == 0:
        print(f'Unable to load VAE model .pth file not found for version {load_version}')
        return None

    encoder_path = sorted(encoder_files)[0]
    decoder_path = sorted(decoder_files)[0]

    print(f"Loading model...")
    print(f"Encoder path -> {encoder_path}")
    print(f"Decoder path -> {decoder_path}")

    encoder = torch.load(encoder_path, map_location=device)
    decoder = torch.load(decoder_path, map_location=device)

    n_in = list(encoder.children())[0].in_features
    n_emb = list(decoder.children())[0].in_features
    n_lay = len([l for l in decoder.children() if type(l) == nn.modules.linear.Linear])

    
    vae =  VAE(n_in, n_emb, n_lay)
    vae.encoder = encoder
    vae.decoder = decoder


    return vae