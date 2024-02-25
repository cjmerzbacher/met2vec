from fluxDataset import FluxDataset, get_conversion_matrix
from vae import FluxVAE
from misc.constants import *

import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def untorch(x : torch.Tensor):
    return x.detach().cpu().numpy()

def get_nonsource(df : pd.DataFrame):
    return df.drop(columns=SOURCE_COLUMNS)

def get_name_prefix(name):
    prefix = f"{name}_" if name != "" else ""
    name = f"{name} " if name != "" else "the "

    return name, prefix

def load_fd(args : argparse.Namespace, name : str = "", seed=None) -> FluxDataset:
    """Load a FluxDataset given its name and args."""
    _, prefix = get_name_prefix(name)

    return FluxDataset(*[args.__dict__[k] for k in [
        f"{prefix}path",
        f"{prefix}size",
        f"{prefix}model_folder",
    ]], seed=seed)
    
def load_multiple_fds(args : argparse.Namespace, name : str = "", seed=None) -> list[FluxDataset]:
    """Load a FluxDataset given its name and args."""
    _, prefix = get_name_prefix(name)

    paths = args.__dict__[f"{prefix}paths"]
    size = args.__dict__[f"{prefix}size"]
    model_folder = args.__dict__[f"{prefix}model_folder"]

    return [
        FluxDataset(path, size, model_folder, seed)
        for path in paths
    ]

def prep_data(data : np.array, preprocessing : str, perplexity : float = 30):
    print(f"Preppring data with {preprocessing}")
    match preprocessing:
        case 'none':
            return data
        case 'tsne':
            tsne = TSNE(perplexity=perplexity)
            return tsne.fit_transform(data)
        case 'pca':
            pca = PCA()
            return pca.fit_transform(data)

def get_fluxes(fd : FluxDataset, join : str):
    if join == INNER:
        return fd.core_reaction_names
    else:
        return fd.reaction_names

def get_data_at_stages(fd : FluxDataset, 
                       vae : FluxVAE, 
                       stages : str = VAE_STAGES, 
                       vae_sample : bool = False,
                       fluxes : list[str] = None,
                       restrictions : dict[str,any] = {}):
    return {
        stage : get_data(fd, vae, stage, vae_sample, fluxes, restrictions)
        for stage in stages
    }

def get_data(fd : FluxDataset, 
             vae : FluxVAE = None, 
             stage : str = EMB, 
             vae_sample : bool = False, 
             fluxes : list[str] = None,
             restrictions : dict[str, any] = {},
             source_columns=[]) -> pd.DataFrame:
    """Transforms the data loaded in a FluxDataset through a vae."""
    if fluxes is None:
        fluxes = fd.reaction_names

    df = fd.get_normalized_data(fluxes)
    
    for column, val in restrictions.items():
        if not column in df.columns:
            print(f"Warning {column} not found in data")
            continue

        df = df[df[column] == val]

    if vae is not None:
        unfufilled_fluxes = set(vae.reaction_names).difference(fluxes)
        if len(unfufilled_fluxes) != 0:
            print(f"Warning VAE used without {len(unfufilled_fluxes)} reqired fluxes!")
    else:
        print("No VAE pressent.")

    if vae is not None and stage != PRE:
        print(f"Using VAE with stage {stage}")

        V = get_nonsource(df).values
        C = get_conversion_matrix(fluxes, vae.reaction_names)
        
        z = vae.encode(V, sample=vae_sample, C=C)

        if stage == EMB:
            columns = [f"emb{i}" for i in range(z.shape[1])]
            data = z
        if stage == REC:
            V_r = vae.decode(z, C, V)
            columns = fluxes
            data = V_r

        df_ext = pd.DataFrame(untorch(data), columns=columns)
        df = pd.concat([df[SOURCE_COLUMNS], df_ext], axis=1)

    df.drop(columns=[c for c in df.columns if c in SOURCE_COLUMNS and c not in source_columns], inplace=True)

    return df