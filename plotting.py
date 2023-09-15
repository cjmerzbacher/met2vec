from plottingDataset import PlottingDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from tqdm import tqdm

import argparse
import random
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams['figure.dpi'] = 60

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["hist", "scatter", "pca_scree"])
    parser.add_argument("--columns", "-c", type=int, nargs=2, default=[0,1])
    parser.add_argument("--preprocessing", "-p", type=str, choices=["normalized", "none", "tsne", "umap", "pca"], default="normalized")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--title", type=str, default="Plot")
    parser.add_argument("datasets", nargs="+", type=str)

    return parser.parse_args()

def expanded_datasets(datasets):
    '''For a list of "dataset" (either files or folders) all folders will be expanded to 
    a list of all .csv files in the folder.'''
    new_datasets = []
    for path in datasets:
        if not path.endswith('.csv'):
            new_datasets += [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        else:
            new_datasets.append(path)
    return new_datasets

def pca(values):
    print("PCA: Fitting...")
    pca = PCA()
    pca.fit(values)
    print("PCA: Transforming...")
    values = pca.transform(values)
    print("PCA: Done.")
    columns = [f"PCA{i}" for i in range(pca.n_components_)]

    return values, columns

def tsne(values, colors, labels, n=2, max_samples=10000):
    print("T-SNE: Fitting...")
    n_samples = min(max_samples, values.shape[0])
    selection = np.random.choice(values.shape[0], n_samples, replace=False)

    values = TSNE(n).fit_transform(values[selection])
    colors = np.array(colors)[selection]
    labels = np.array(labels)[selection]

    print("T-SNE: Done...")
    columns = [f"E{i}" for i in range(n)] 
    return values, columns, colors, labels

def umap(values):
    print("UMAP: Fitting...")
    values = UMAP(min_dist=0.85).fit_transform(values)
    print("UMAP: Done.")
    columns = [f"E{i}" for i in range(2)] 

    return values, columns


def preprocess(plotD, preprocessing):
    '''Preprocesses the data in different ways depending on the preprocessing arg passed in.
    
    Returns values, columns, colors, labels to be used in a scatter plot by default.'''
    values = plotD.normalized_values()
    columns = plotD.columns()
    colors = plotD.colors()
    labels = plotD.labels()

    match preprocessing:
        case "pca":
            values, columns = pca(values)
        case "tsne":
            values, columns, colors, labels = tsne(values, colors, labels)
        case "none":
            values = plotD.values()
        case "normalized":
            values = plotD.normalized_values()
        case "umap":
            values, columns = umap(values)

    return values, columns, colors, labels

#
#   PLOTTING
#

def hists(values, column_names, labels, n=8):
    '''Takes a grid of values, with coresponding column names and labels and prints and nxn 
    grid of histograms for n^2 in total, each for a different columns of the values passed in.'''
    fig, axs = plt.subplots(n, n)

    if values.shape[1] < n*n:
        print(f"Not enough rows for {n}*{n} grid, quiting.")
        quit()

    zeros = values == 0
    possible_plots = [i for i in range(values.shape[1] - 1) if not all(zeros[:,i])]
    random.shuffle(possible_plots)

    for i in range(0, n*n):
        ax = axs[i%n, int(i/n)]
        column = values[:, possible_plots[i]]
        unique_labels = list(set(labels))
        labeled_data = [column[np.array(labels) == label] for label in unique_labels]
        ax.hist(labeled_data, bins=40, stacked=True, label=unique_labels)

        title = column_names[i]
        if len(title) > 30:
            title = title[0:28] + "..."
        ax.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(100)
    fig.set_figwidth(100)

def scatter(values, columns, column_names, colors, handles):
    '''Takes values, columns, column_names, colors and handles and plots a scatter plot.'''
    x_ax, y_ax = columns
    x = values[:, x_ax]
    y = values[:, y_ax]

    plt.scatter(x, y, c=colors)
    plt.legend(handles=handles)
    plt.xlabel(column_names[x_ax])
    plt.ylabel(column_names[y_ax])

def pca_scree(values):
    '''Takes values, runs a pca and print the coresponding scree plot.'''
    print("PCA Scree: Fitting...")
    pca = PCA()
    pca.fit(values)
    print("PCA Scree: Done.")

    evr = pca.explained_variance_ratio_

    plt.plot(evr, label='Explained Variance')
    plt.plot([sum(evr[:i]) for i in range(pca.n_components_)], label="Cumulative Variace")

    plt.xlabel("Principal Components")
    plt.ylabel("% Total Variance")
    plt.legend()

#
#   MAIN
#

def main():
    args = get_args()
    plotD = PlottingDataset()
    for path in tqdm(expanded_datasets(args.datasets), desc="Loading datasets"):
        plotD.add_section(path)

    values, columns, colors, labels = preprocess(plotD, args.preprocessing)

    match args.type:
        case "hist":
            hists(values, columns, labels)
        case "scatter":
            scatter(values, args.columns, columns, colors, plotD.handles())
        case "pca_scree":
            pca_scree(values)

    plt.suptitle(args.title, fontsize=32)
    plt.show()

if __name__ == "__main__":
    main()