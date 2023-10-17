import pandas as pd
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt
import torch

from fluxDataset import FluxDataset
from vae import make_VAE_from_args, VAE
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PLOT_CONFIG_PATH = '.plotconfig.json'
LABLE_CONFIG = 'lable_config'


def get_args():
    parser = argparse.ArgumentParser('Flux Plotter')
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    parser.add_argument('dataset')
    parser.add_argument('-n', '--dataset_size', type=int, default=1024)
    parser.add_argument('-j', '--dataset_join', choices=['inner', 'outer'], default='inner')
    parser.add_argument('-r', '--dataset_reload_aux', type=bool, default=False)
    parser.add_argument('-e', '--encoder')
    parser.add_argument('-p', '--preprocessing', choices=['none', 'tsne', 'pca'], default='none')
    parser.add_argument('--perplexity', type=float, default=30.0)

    parser.add_argument('-s', '--save_plot')
    parser.add_argument('-t', '--title', default="")

    args = parser.parse_args()
    args.folder = os.path.dirname(args.dataset)
    args.plot_config_path = os.path.join(args.folder, PLOT_CONFIG_PATH)

    return args

def load_plot_config(fd : FluxDataset, args):
    plot_config = {LABLE_CONFIG : {}}
    try:
        with open(args.plot_config_path, 'r') as plot_config_file:
            plot_config = json.load(plot_config_file)
    except:
        pass

    for name in fd.data['label'].unique():
        if not name in plot_config[LABLE_CONFIG]:
            plot_config[LABLE_CONFIG][name] = {}
        config = plot_config[LABLE_CONFIG][name]
        if 'color' not in config:
            config['color'] = '#FF00FF'
        if 'marker' not in config:
            config['marker'] = 'o'
        if 'label' not in config:
            config['label'] = name

    if not 'dpi' in plot_config:
        plot_config['dpi'] = 100 
    if not 'figsize' in plot_config:
        plot_config['figsize'] = (10, 8) 
    if not 'dot_size' in plot_config:
        plot_config['dot_size'] = 1.0
    if not 'ldot_size' in plot_config:
        plot_config['ldot_size'] = 30.0
    if not 'lfontsize' in plot_config:
        plot_config['lfontsize'] = 8.0
    if not 'lbbox' in plot_config:
        plot_config['lbbox'] = None
    if not 'plot_width' in plot_config:
        plot_config['plot_width'] = 1.0

    with open(args.plot_config_path, 'w') as plot_config_file:
        json.dump(plot_config, plot_config_file, indent=4)

    return plot_config

def load_encoder(fd, args):
    if args.encoder is None:
        return

    vae = make_VAE_from_args(fd.data.shape[1] - 1, os.path.join(os.path.dirname(args.encoder), 'args.json'))
    vae.encoder.load_state_dict(torch.load(args.encoder).state_dict())

    return vae.encoder


def main():
    args = get_args()
    fd = FluxDataset(args.dataset, args.dataset_size, args.dataset_join, args.verbose, args.dataset_reload_aux)
    plot_config = load_plot_config(fd, args)
    encoder = load_encoder(fd, args)

    plt.figure(figsize=plot_config['figsize'])
    ax = plt.subplot(111)

    data = fd.data.drop(columns='label').values
    if encoder is not None:
        data = encoder(torch.Tensor(data)).detach().cpu().numpy()
        data = data[:,:data.shape[1] // 2]

    match args.preprocessing:
        case 'none':
            pass
        case 'tsne':
            print('Fitting tsne...')
            tsne = TSNE(perplexity=args.perplexity)
            data = tsne.fit_transform(data)
        case 'pca':
            print("Fitting PCA...")
            pca = PCA()
            data = pca.fit_transform(data)

    for name in tqdm(fd.data['label'].unique(), disable=not args.verbose, desc='Plottig data'):
        config = plot_config[LABLE_CONFIG][name]
        label_data = data[fd.data['label'] == name]

        ax.scatter(label_data[:,0], label_data[:,1],
                    color=config['color'],
                    marker=config['marker'],
                    label=config['label'],
                    s=plot_config['dot_size'])
    
    plt.title = args.title
    legend = plt.legend(fontsize=plot_config['lfontsize'], bbox_to_anchor=plot_config['lbbox'])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * plot_config['plot_width'], box.height])

    for handle in legend.legend_handles:
        handle.set_sizes([plot_config['ldot_size']])

    if args.save_plot is None:
        plt.show()
    else:
        plt.savefig(args.save_plot, dpi=plot_config['dpi'])
    

if __name__ == '__main__':
    main()