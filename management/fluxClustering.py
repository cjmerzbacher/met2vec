
from sklearn.cluster import KMeans, DBSCAN
from vae import VAE

from constants import *
from fluxDataset import FluxDataset, get_data

def get_clustering(fd : FluxDataset,  clustering_type, vae : VAE = None, vae_stage : str = EMB, vae_sample=True, dbscan_params=None) -> list:
    data = get_data(fd, vae_stage, vae, vae_sample)

    labels = fd.data['label'].unique()
    match clustering_type:
        case "none":
            return [sample['label'] for sample in fd.data.iloc]
        case "kmeans":
            kmeans = KMeans(len(labels), n_init='auto').fit(data)
            return kmeans.labels_
        case "dbscan":
            eps, mins = dbscan_params[vae_stage] if dbscan_params != None else (1.0, 5)
            dbscan = DBSCAN(eps=eps, min_samples=mins).fit(data)
            return dbscan.labels_