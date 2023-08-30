from plottingDataset import PlottingDataset
from tqdm import tqdm

import argparse
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 60

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["hist"])
    parser.add_argument("datasets", nargs="+", type=str)

    return parser.parse_args()

def hist(plotD):
    n = 8
    fig, axs = plt.subplots(n, n)
    normalized_values = plotD.normalized_values()

    zeros = plotD.values() == 0
    possible_plots = [i for i in range(plotD.shape()[1] - 1) if not all(zeros[:,i])]
    random.shuffle(possible_plots)

    for i in range(0, n*n):
        ax = axs[i%n, int(i/n)]
        ax.hist(normalized_values[:,possible_plots[i]], bins=40)

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

args = get_args()
plotD = PlottingDataset()
for path in tqdm(args.datasets, desc="Loading datasets"):
    plotD.add_section(path)

match args.type:
    case "hist":
        hist(plotD)