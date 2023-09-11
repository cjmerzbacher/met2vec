from plottingDataset import PlottingDataset
from sklearn.decomposition import PCA
from tqdm import tqdm

import argparse
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 60

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["hist", "scatter"])
    parser.add_argument("-a", "--axis", type=int, nargs=2, default=[0,1])
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("datasets", nargs="+", type=str)

    return parser.parse_args()

def expanded_datasets(args):
    return args.datasets

def hist(plotD, values):
    n = 8
    fig, axs = plt.subplots(n, n)

    zeros = plotD.values() == 0
    possible_plots = [i for i in range(plotD.shape()[1] - 1) if not all(zeros[:,i])]
    random.shuffle(possible_plots)

    for i in range(0, n*n):
        ax = axs[i%n, int(i/n)]
        ax.hist(values[:,possible_plots[i]], bins=40, stacked=True, label=plotD.labels())

        title = plotD.columns()[i]
        if len(title) > 30:
            title = title[0:28] + "..."
        ax.set_title(title)

    fig.suptitle('Flux Distributions', fontsize=32)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(100)
    fig.set_figwidth(100)
    plt.show()

def scatter(plotD, values, args):
    x_ax, y_ax = args.axis
    x = values[:, x_ax]
    y = values[:, y_ax]

    plt.scatter(x, y, c=plotD.colors())
    plt.legend(handles=plotD.handles())
    plt.xlabel(x_ax)
    plt.ylabel(y_ax)
    plt.show()

args = get_args()
plotD = PlottingDataset()
for path in tqdm(expanded_datasets(args), desc="Loading datasets"):
    plotD.add_section(path)

values = plotD.normalized_values()

if args.pca == True:
    print("Fitting PCA...")
    pca = PCA()
    pca.fit(values)
    values = pca.transform(values)

match args.type:
    case "hist":
        hist(plotD, values)
    case "scatter":
        scatter(plotD, values, args)