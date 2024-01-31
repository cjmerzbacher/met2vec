from fluxDataset import FluxDataset
from .fluxDataset import get_data
from vae import FluxVAE
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

def train_gmms(fd : FluxDataset, vae : FluxVAE, vae_stage : str, vae_sample : bool, k : int = 1, v : bool = True) -> dict[str,GaussianMixture]:
    labels = set(fd.labels)
    gmms = {label : GaussianMixture(k) for label in labels}

    for label, gmm in tqdm(gmms.items(), desc="Training gmms", disable=not v, position=0):
        train_data = get_data(fd, vae, vae_stage, vae_sample, label) 
        gmm.fit(train_data)

    return gmms
    