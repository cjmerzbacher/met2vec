from sklearn.cluster import KMeans

from misc.constants import *

import pandas as pd
import numpy as np

def get_flux_kmeans(k, data : pd.DataFrame):
    data = data.drop(columns=SOURCE_COLUMNS)
    return KMeans(k, n_init='auto').fit(data)

def get_KMeans_classifications(k, n, data : pd.DataFrame):
    classifications = []
    for _ in range(n):
        kmeans = get_flux_kmeans(k, data)
        classifications.append(kmeans.labels_)

    return classifications

def get_kmeans_inertia(k, data : pd.DataFrame):
    return get_flux_kmeans(k, data).inertia_

def get_k(args, fd, origional_clustering=LABEL):
    k = args.k
    if k == None:
        if origional_clustering in fd.data.columns:
            k = fd.data[origional_clustering].nunique()
    return k

def get_max_k(args, fd):
    max_k = args.max_k
    if max_k is None:
        if FILE_N in fd.data.columns:
            max_k = fd.data[FILE_N].nunique()
        else:
            print(f"{FILE_N} not in fd.data.columns using max_k=1!")
            max_k = 1
    return max_k