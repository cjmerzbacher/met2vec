import pandas as pd
import matplotlib.pyplot as plt
import os

from pandas import DataFrame
from sklearn.decomposition import PCA
from pprint import pp

def load_flux_dataset(file):
    print(f"Loading {file} ...")
    df = pd.read_csv(file)
    df.drop(columns=df.columns[0], inplace=True)
    df['label'] = 0

    return df

def load_flux_datasets(folder):
    files = [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith('.csv')]
    datasets = map(load_flux_dataset, files)
    datasets = map(set_label, enumerate(datasets))
    combined_df = pd.concat(datasets)
    combined_df.fillna(0, inplace=True)

    return combined_df

def load_flux_data(path : str):
    if path.endswith(".csv"):
        return load_flux_dataset(path)
    return load_flux_datasets(path)

def normalize_flux_data(dataset : DataFrame):
    all_zeros = (dataset == 0).all()
    zero_columns = [n for n in dataset.columns if all_zeros[n]]
    mean = dataset.mean()
    std = dataset.std()
    std[zero_columns] = 1.0

    return (dataset - mean) / std

def set_label(idf):
    i, df = idf
    df['label'] = i
    return df

def pca(dataset : DataFrame):
    print(f"Performing PCA on dataset of size {dataset.shape}...")
    pca = PCA()
    pca.fit(dataset.drop(columns=['label'], inplace=False).values)
    return pca

def plot_explained_variance_ratio(flux_pca : PCA):
    plt.plot(flux_pca.explained_variance_ratio_)
    plt.show()

def plot_scatter(x, y, labels):
    unique_labels = list(set(labels))
    colors = ['r', 'g', 'b']
    plt.scatter(x, y, c=[colors[unique_labels.index(l)] for l in labels])
    plt.show()

def plot_hist(values, bins=40):
    plt.hist(values, bins=bins)
    plt.show()


flux_df = load_flux_data("./data/samples/test/") 
pp(flux_df)
flux_df_normalized = normalize_flux_data(flux_df)

plot_hist(flux_df_normalized.values[:,7])

flux_pca = pca(flux_df_normalized)

flux_vals_transformed = flux_pca.transform(flux_df_normalized.drop(columns=['label'], inplace=False).values)

plot_scatter(flux_vals_transformed[:,0], flux_vals_transformed[:,1], flux_df['label'])

