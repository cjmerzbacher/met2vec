from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_save_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save_path", help="Where output will be saved.")

    return parser

def get_title_parser(name: str = None, default : str = None):
    name = f"{name}_" if name != None else ""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(f"--{name}title", default=default)

    return parser

def get_score_distribution(X, score = adjusted_rand_score):
    n = len(X)
    scores = np.zeros((n, n))
    std = np.zeros((n, n))
    for (i, j) in tqdm([(i, j) for i in range(n) for j in range(n)], desc='Calculating dist'):
        combinations = [(X[i], X[j])] if type(X[i]) != list else [(x1, x2) for x1 in X[i] for x2 in X[j]]
        score_dist = [score(s1, s2) for s1, s2 in combinations]
        scores[i,j] = np.mean(score_dist)
        std[i,j] = np.std(score_dist)

    return scores, std

def plot_comparison(ax, scores, names_y, names_x, std = None, write_scores = True, y_label="", x_label=""):
    im = ax.imshow(scores)

    ax.set_xticks(np.arange(scores.shape[1]), labels=names_x)
    ax.set_yticks(np.arange(scores.shape[0]), labels=names_y)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if write_scores:
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                text =f'{scores[i,j]:.2f}'
                if std != None and std[i,j] > 0.005:
                    text += f'$\pm${std[i,j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', color='w')
    else:
        plt.colorbar(im)