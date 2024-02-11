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
        f"{prefix}model_folder",
    ]], seed=seed)
    
    
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
             restrictions : dict[str, any] = {}) -> np.array:
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

    return df