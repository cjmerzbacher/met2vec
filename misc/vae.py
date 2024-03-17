import os
import json
import torch
import torch.nn as nn

from argparse import Namespace
from vae import FluxVAE
from fluxDataset import FluxDataset

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
    return args.vae_folder, args.vae_version, args.legacy_vae, args.legacy_train_folder, args.legacy_model_folder

def load_v_mu_and_v_std_from_train_folder(train_folder, model_folder, reaction_names):
    if train_folder is None:
        print("No train folder given!")
        quit()

    path = os.path.join(train_folder, MU_STD_FILE)
    v_mu_v_std = safe_json_load(path)
    
    if v_mu_v_std is not None:
        v_mu = v_mu_v_std[V_MU]
        v_std = v_mu_v_std[V_STD]
    else:
        print(f"'{MU_STD_FILE}' not found. Loading using FluxDataset.")
        fd = FluxDataset(train_folder, 1024, model_folder, seed=0)
        v_mu, v_std = fd.get_mu_std_for_reactions(reaction_names)

        if v_mu is None or v_std is None:
            print("Unable to load v_mu and v_std, exiting!")
            quit()

        safe_json_dump(path, {
            V_MU : v_mu.tolist(),
            V_STD : v_std.tolist()
        }, True)

    return v_mu, v_std

def load_VAE(
        folder, 
        version=None, 
        legacy_vae=False,
        legacy_train_folder=None,
        legacy_model_folder=None
        ) -> FluxVAE:
    """Loads a VAE from a given folder."""

    def contains_correct_version(f : str, model_part : str):
        return model_part in f and (True if version is None else str(version) in f)
    
    encoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'encoder')] + [""]
    decoder_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'decoder')] + [""]
    desc_files = [os.path.join(folder, f) for f in os.listdir(folder) if contains_correct_version(f, 'vae_desc')] + [""]


    if len(encoder_files) == 0 or len(decoder_files) == 0:
        print(f'Unable to load VAE model .pth file not found for version {version}')
        return None

    encoder_path = sorted(encoder_files)[-1]
    decoder_path = sorted(decoder_files)[-1]
    desc_path = sorted(desc_files)[-1]

    print(f"Loading model...")
    print(f"Encoder path -> {encoder_path}")
    print(f"Decoder path -> {decoder_path}")
    print(f"Desc path    -> {desc_path}")

    encoder = torch.load(encoder_path, map_location=device)
    decoder = torch.load(decoder_path, map_location=device)

    n_in = list(encoder.children())[0].in_features
    n_emb = list(decoder.children())[0].in_features
    n_lay = len([l for l in decoder.children() if type(l) == nn.modules.linear.Linear])

    vae_args = read_VAE_args(os.path.join(folder, ARGS_PATH))
    vae_desc = read_VAE_decriptor(desc_path)

    reaction_names = safe_extract_from_desc(vae_desc, "reaction_names", None)
    weight_decay = safe_extract_from_desc(vae_desc, "weight_decay", 0.0) 

    v_mu = safe_extract_from_desc(vae_desc, "v_mu", None)
    v_std = safe_extract_from_desc(vae_desc, "v_std", None)

    if v_mu is None or v_std is None:
        print(f"v_mu or v_std not pressent in VAE desc, loading from dataset...")
        print(f"Using legacy train folder '{legacy_train_folder}' with model folder '{legacy_model_folder}'.")
        v_mu, v_std = load_v_mu_and_v_std_from_train_folder(
            legacy_train_folder, 
            legacy_model_folder,
            reaction_names)

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
        reaction_names=reaction_names,
        v_mu=v_mu,
        v_std=v_std,
    )
    vae.encoder = encoder
    vae.decoder = decoder

    return vae