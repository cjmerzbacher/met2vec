from sklearn.cluster import KMeans

def get_KMeans_classifications(k, n, data):
    classifications = []
    for _ in range(n):
        kmeans = KMeans(k, n_init='auto').fit(data)
        classifications.append(kmeans.labels_)

    return classifications