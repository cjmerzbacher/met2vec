import os
import torch
import torch.nn as nn
import argparse

from vae import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_VAE(args) -> VAE:
    """Loads a VAE from a given folder. 
    
    The folder should contain encoder{version}.pth 
    and decoder{version}.pth files. If no load_version is given the highest load
    version found will be used.

    Args:
        args: The args namespae containing vae_folder and vae_version.

    Returns:
        vae: A VAE with weights loaded from the specified folder.
    """

    folder = args.vae_folder
    load_version = args.vae_version

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