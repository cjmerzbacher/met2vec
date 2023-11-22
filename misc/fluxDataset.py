from fluxDataset import FluxDataset
from vae import VAE
from misc.constants import *
from .parsing import boolean_string

import argparse
import numpy as np

def get_name_prefix(name):
    prefix = f"{name}_" if name != "" else ""
    name = f"{name} " if name != "" else "the "

    return name, prefix

def make_load_fluxDataset_parser(name : str = "", path_tag=None):
    name, prefix = get_name_prefix(name)
    path_tag = [path_tag] if path_tag != None else []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*path_tag, f"--{prefix}path", required=True, help=f"The path {name}dataset will be laoded from.")
    parser.add_argument(f"--{prefix}size", default=65536, type=int, help=f"The size of {name}dataset samples.")
    parser.add_argument(f"--{prefix}test_size", default=0, type=int, help=f"The size of the test set for {name}dataset.")
    parser.add_argument(f"--{prefix}join", default=INNER, choices=DATASET_JOINS, help=f"The join that will be used for {name}dataset.")
    parser.add_argument(f"--{prefix}reload_aux", type=boolean_string, default=True, help=f"Whether {name}dataset should reload aux.")
    parser.add_argument(f"--{prefix}skip_tmp", type=boolean_string, default=False, help=f"Whether {name}dataset should reload the tmp files.")

    return parser

def load_fd(args : argparse.Namespace, name : str) -> FluxDataset:
    """Load a FluxDataset given its name and args.
    
    Arguments:
        args: The arguments namespace
        name: The name of the dataset

    Returns:
        fd: The FluxDataset
    """
    _, prefix = get_name_prefix(name)

    return FluxDataset(*[args.__dict__[k] for k in [
        f"{prefix}path",
        f"{prefix}size",
        f"{prefix}test_size",
        f"{prefix}join",
        f"{prefix}reload_aux",
        f"{prefix}skip_tmp",
    ]])
    
    

def get_data(fd : FluxDataset, vae : VAE = None, stage : str = EMB, vae_sample : bool = False, label : str = None) -> np.array:
    """Transforms the data loaded in a FluxDataset through a vae.
     
    Transform the sample loaded into a FluxDataset possibly restricted to a sample. The sample will be left 
    unchanged, transformed into the VAE embedding, or reconstructed from it's VAE embedding.
    
    Arguments:
        fd: The FluxDataset whose sample will be transformed.
        vae: The VAE which will be used to transform the data.
        stage: The stage in the VAE which be output ('pre', 'emb', 'post').
        vae_sample: If true the VAE will sample from the embedding distribution, isntead of using the mean.
        label: The label for the subset of the FluxDataset that will be transformed. By default the whole
            sample will be used.

    Return:
        data: The transformed subset of the FluxDataset sample.
    """
    if label is None:
        data = fd.values
    else:
        data = fd[label]

    if vae and stage != PRE:
        data = vae.encode(data, sample=vae_sample)
        if stage == REC:
            data = vae.decode(data)
        data = data.detach().cpu().numpy()
    return data