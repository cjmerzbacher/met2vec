from sklearn.cluster import KMeans

from misc.constants import *

import pandas as pd

def get_KMeans_classifications(k, n, data : pd.DataFrame):
    data = data.drop(columns=SOURCE_COLUMNS)

    classifications = []
    for _ in range(n):
        kmeans = KMeans(k, n_init='auto').fit(data)
        classifications.append(kmeans.labels_)

    return classifications

def get_k(args, fd):
    k = args.k
    if k == None:
        k = len(fd.unique_labels)
    return k