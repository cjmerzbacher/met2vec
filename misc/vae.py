import os
import json
import torch
import torch.nn as nn

from argparse import Namespace
from vae import FluxVAE

from misc.constants import *
from misc.io import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_VAE_args(path) -> dict[str, str]:
    return safe_json_load(path)

def read_VAE_decriptor(path):
    return safe_json_load(path)

def safe_extract_from_args(args : Namespace, name : str, default : any):
    if hasattr(args, name):
        return vars(args)[name]
    return default

def safe_extract_from_desc(desc : dict[str, any], name : str, default : any):
    if desc is not None:
        if name in desc:
            return desc[name]
    return default

def get_load_VAE_args(args):
    return args.vae_folder, args.vae_version, args.legacy_vae

def load_VAE(folder, version=None, legacy_vae=False) -> FluxVAE:
    """Loads a VAE from a given folder."""

    def contains_correct_version(f : str, model_part : str):
        return model_part in f and (True if version is None else str(version) in f)
    
    encoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'encoder')] + [""]
    decoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'decoder')] + [""]
    desc_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'vae_desc')] + [""]
    vae_pkl_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'vae_pkl')] + [""]

    if len(encoder_files) == 0 or len(decoder_files) == 0:
        print(f'Unable to load VAE model .pth file not found for version {version}')
        return None

    encoder_path = sorted(encoder_files)[-1]
    decoder_path = sorted(decoder_files)[-1]
    desc_path = sorted(desc_files)[-1]
    vae_pkl_path = sorted(vae_pkl_files)[-1]

    print(f"Loading model...")
    print(f"VAE pkl path -> {vae_pkl_path}")
    print(f"Encoder path -> {encoder_path}")
    print(f"Decoder path -> {decoder_path}")
    print(f"Desc path    -> {desc_path}")

    vae = safe_pkl_load(vae_pkl_path)
    if vae is not None:
        print(f"pkl loaded!")
        return vae

    encoder = torch.load(encoder_path, map_location=device)
    decoder = torch.load(decoder_path, map_location=device)

    n_in = list(encoder.children())[0].in_features
    n_emb = list(decoder.children())[0].in_features
    n_lay = len([l for l in decoder.children() if type(l) == nn.modules.linear.Linear])

    vae_args = read_VAE_args(os.path.join(folder, ARGS_PATH))
    vae_desc = read_VAE_decriptor(desc_path)

    reaction_names = safe_extract_from_desc(vae_desc, "reaction_names", None)
    weight_decay = safe_extract_from_desc(vae_desc, "weight_decay", 0.0) 

    lrelu_slope = safe_extract_from_args(vae_args, "lrelu_slope", 0.0)
    batch_norm = safe_extract_from_args(vae_args, "batch_norm", False)
    dropout_p = safe_extract_from_args(vae_args, "dropout", 0.0)

    vae =  FluxVAE(
        n_in=n_in, 
        n_emb=n_emb, 
        n_lay=n_lay, 
        lrelu_slope=lrelu_slope, 
        batch_norm=batch_norm, 
        dropout_p=dropout_p, 
        legacy_vae=legacy_vae, 
        weight_decay=weight_decay, 
        reaction_names=reaction_names
    )
    vae.encoder = encoder
    vae.decoder = decoder

    return vae