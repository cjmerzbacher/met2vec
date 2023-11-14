import torch
import os
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def format_input(x):
    return torch.Tensor(x).to(device)

def get_linear_network(n_in : int, n_out : int, n_lay : int, lrelu_slope : float, batch_norm : bool, dropout_p : float) -> nn.Module:
    model = []
    sizes = [round(s) for s in np.linspace(n_in, n_out, n_lay)]
    for v_in, v_out in zip(sizes[0:-2], sizes[1:-1]):
        model += [nn.Linear(v_in, v_out), nn.LeakyReLU(negative_slope=lrelu_slope), nn.Dropout(dropout_p)]
        if batch_norm: model += [nn.BatchNorm1d(v_out)]
    model += [nn.Linear(sizes[-2], sizes[-1])]
    model = nn.Sequential(*model).float()
    model.to(device)
    return model


class VAE:
    def __init__(self, 
                 n_in : int, 
                 n_emb : int, 
                 n_lay : int, 
                 lrelu_slope : float = 0.01, 
                 batch_norm : bool = False, 
                 dropout_p : float = 0.0
                 ):
        """Initializes a VAE with the dimensions and hyperparameters given.

        This will construct a VAE with the dimensions and hyperparameters given.
        For different layers the intermediate layer dimensions are a linear function
        from the encoder / decoder input size to their respective output sizes.

        Args:
            n_in: The number of input / reconstruction dimensions.
            n_emb: The number of dimensions for the embedding space.
            n_lay: The number of layers in the encoder and decoder.
            lrelu_slope: The leaky-ReLU slope used.
            batch_norm: If true batch_norm will be applied between the layers of
            the encoder and decoder.
            dropout_p: The dropout percentage used.
        """

        self.n_in = n_in
        self.n_emb = n_emb
        self.n_lay = n_lay

        self.encoder = get_linear_network(n_in, n_emb * 2, n_lay, lrelu_slope, batch_norm, dropout_p)
        self.decoder = get_linear_network(n_emb, n_in, n_lay, lrelu_slope, batch_norm, dropout_p)

    def get_dist(self, x : torch.Tensor) -> torch.Tensor:
        """Gets the distribution values for a given input value.

        The distirbution $P(z\mid x)$ is given as a parameterization for a
        multivariate gaussian distribution. This is the output from the encoder.
        
        Args:
            x: The input to the encoder.

        Returns:
            mu: The mean of the distirbution.
            sigma: The diagonals for the distirbution covariance matrix.
        """
        x = format_input(x)
        y = self.encoder(x)
        self.mu = y[:,:self.n_emb]
        self.sigma = torch.log(1.0 + torch.exp(y[:,self.n_emb:])) #Soft plus

        return self.mu, self.sigma
    
    def encode(self, x : torch.Tensor, sample : bool = True) -> torch.Tensor:
        """Encodes an embedding into the latent space.

        Args: 
            x: The input to be encoded.

        Returns:
            z: The encoding of the input.
        """
        x = format_input(x)

        mu, sigma = self.get_dist(x)

        ones = torch.ones_like(sigma)
        epsilon = torch.normal(ones, ones * 0).to(device)

        z = mu + ((sigma * epsilon) if sample else 0.0)

        return z
    
    def decode(self, z : torch.Tensor) -> torch.Tensor:
        """Decodes a embedding / reconstructs the value embeded.
        
        Args:
            z: A point in the latent space / embedding space.
            
        Returns:
            y: A reconstruciton / decoding of the latent space point.
        """
        z = format_input(z)
        y = self.decoder(z)
        return y


    def encode_decode(self, x : torch.Tensor) -> torch.Tensor:
        """Runs a sample through the network encoding and then decoding it without sampling.
        
        Args:
            x: The input which will be fed into the encoder.
            
        Returns:
            y: The reconstruciton from the decoder.
        """
        x = format_input(x)

        z = self.encode(x)
        y = self.decode(z)
        return y

    def loss(self, x : torch.Tensor, y : torch.Tensor) -> tuple[torch.Tensor, dict[str,float]]:
        """Computes the loss for a given x and y.
        
        Note x and y should be made by this model, otherwise autograd will not
        apply gradient properly.

        Args:
            x: The x value which the network took as input for the encoder.
            y: The reconstruciton made by the decoder.

        Returns:
            loss: The overall loss from the network (torch.Tensor).
            blame: A dictionary describing how different parts of the 
            loss contributed to the overall loss.
        """
        x = format_input(x)
        y = format_input(y)

        # Reconstruction loss
        loss_reconstruction = torch.sum(torch.pow(x - y, 2.0), dim=1) 
        loss_reconstruction = torch.mean(loss_reconstruction)

        # Divergence from N(0, 1)
        loss_divergence = 0.5 * torch.sum(self.sigma, dim=1)                      # tr(sigma)
        loss_divergence += 0.5 * torch.norm(self.mu, dim=1)                       # mu^T @ mu
        loss_divergence -= 0.5 * torch.sum(torch.log(self.sigma + 0.0001), dim=1) #log(sigma)
        loss_divergence = torch.mean(loss_divergence)

        loss = loss_reconstruction + loss_divergence

        blame = {
            "loss" : loss.numpy(),
            "loss_divergence" : loss_divergence.numpy(),
            "loss_reconstruction" : loss_reconstruction.numpy()
        }
         
        return loss, blame
    

def load_VAE(folder : str, load_version : int = None) -> VAE:
    """Loads a VAE from a given folder. 
    
    The folder should contain encoder{version}.pth 
    and decoder{version}.pth files. If no load_version is given the highest load
    version found will be used.

    Args:
        folder: The folder the VAE will be loaded from
        load_versio: The version of the VAE that will be loaded.

    Returns:
        vae: A VAE with weights loaded from the specified folder.
    """
    def contains_correct_version(f : str, model_part : str):
        return model_part in f and (True if load_version is None else str(load_version) in f)
    
    encoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'encoder')]
    decoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'decoder')]

    if len(encoder_files) == 0 or len(decoder_files) == 0:
        print(f'Unable to load VAE model .pth file not found for version {load_version}')
        return None

    encoder_path = sorted(encoder_files)[-1]
    decoder_path = sorted(decoder_files)[-1]

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