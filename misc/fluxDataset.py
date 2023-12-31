from fluxDataset import FluxDataset
from vae import VAE
from misc.constants import *

import argparse
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
        f"{prefix}test_size",
        f"{prefix}join",
        f"{prefix}verbose",
        f"{prefix}reload_aux",
        f"{prefix}skip_tmp",
    ]], seed=seed)
    
    
def prep_data(data : np.array, preprocessing : str, perplexity : float = 30):
    match preprocessing:
        case 'none':
            return data
        case 'tsne':
            tsne = TSNE(perplexity=perplexity)
            return tsne.fit_transform(data)
        case 'pca':
            pca = PCA()
            return pca.fit_transform(data)

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
        data = fd.normalized_values
    else:
        data = fd[label]

    if vae and stage != PRE:
        data = vae.encode(data, sample=vae_sample)
        if stage == REC:
            data = vae.decode(data)
        data = data.detach().cpu().numpy()
    return data