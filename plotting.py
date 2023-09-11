from plottingDataset import PlottingDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--title", type=str, default="Plot")
    parser.add_argument("datasets", nargs="+", type=str)

    return parser.parse_args()

def expanded_datasets(args):
    datasets = []
    for path in args.datasets:
        if not path.endswith('.csv'):
            datasets += [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        else:
            datasets.append(path)
    return datasets

def hist(values, column_names):
    n = 8
    fig, axs = plt.subplots(n, n)

    zeros = values == 0
    possible_plots = [i for i in range(values.shape[1] - 1) if not all(zeros[:,i])]
    random.shuffle(possible_plots)

    for i in range(0, n*n):
        ax = axs[i%n, int(i/n)]
        ax.hist(values[:,possible_plots[i]], bins=40, stacked=True, label=plotD.labels())

        title = column_names[i]
        if len(title) > 30:
            title = title[0:28] + "..."
        ax.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(100)
    fig.set_figwidth(100)

def scatter(values, columns, column_names, colors, handles):
    x_ax, y_ax = columns
    x = values[:, x_ax]
    y = values[:, y_ax]

    plt.scatter(x, y, c=colors)
    plt.legend(handles=plotD.handles())
    plt.xlabel(column_names[x_ax])
    plt.ylabel(column_names[y_ax])

def pca_scree(values):
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


args = get_args()
plotD = PlottingDataset()
for path in tqdm(expanded_datasets(args), desc="Loading datasets"):
    plotD.add_section(path)

values = plotD.normalized_values()
columns = plotD.columns()
colors = plotD.colors()

if args.pca == True:
    print("PCA: Fitting...")
    pca = PCA()
    pca.fit(values)
    print("PCA: Transforming...")
    values = pca.transform(values)
    print("PCA: Done.")
    columns = [f"PCA{i}" for i in range(pca.n_components_)]

if args.tsne == True:
    print("T-SNE: Fitting...")
    n = 2
    n_samples = min(6000, values.shape[0])
    selection = np.random.choice(values.shape[0], n_samples, replace=False)

    values = TSNE(n).fit_transform(values[selection])
    colors = np.array(colors)[selection]

    print("T-SNE: Done...")
    columns = [f"E{i}" for i in range(n)] 

match args.type:
    case "hist":
        hist(values, columns)
    case "scatter":
        scatter(values, args.columns, columns, colors, plotD.handles())
    case "pca_scree":
        pca_scree(values)

plt.suptitle(args.title, fontsize=32)
plt.show()