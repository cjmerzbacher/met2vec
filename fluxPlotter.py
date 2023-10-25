import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from fluxDataset import FluxDataset
from vae import make_VAE, VAE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PLOT_CONFIG_PATH = '.plotconfig.json'
LABEL_CONFIG = 'lable_config'


def get_args():
    parser = argparse.ArgumentParser('Flux Plotter')
    parser.add_argument('dataset')
    parser.add_subparsers(dest='command')
    
    parser.add_argument('-n', '--dataset_size', type=int, default=1024)
    parser.add_argument('-j', '--dataset_join', choices=['inner', 'outer'], default='inner')
    parser.add_argument('-r', '--dataset_reload_aux', type=bool, default=False)
    parser.add_argument('-v', '--vae_folder')
    parser.add_argument('--vae_version', type=int)
    parser.add_argument('--vae_sample', type=bool, default=False)


    parser.add_argument('-p', '--preprocessing', choices=['none', 'tsne', 'pca'], default='none')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--clustering', choices=['none', 'kmeans', 'dbscan'], default='none')
    parser.add_argument('--cluster_after_vae', type=bool, default=True)

    parser.add_argument('-s', '--save_plot')
    parser.add_argument('-t', '--title', default="")

    args = parser.parse_args()
    args.folder = os.path.dirname(args.dataset)
    args.plot_config_path = os.path.join(args.folder, PLOT_CONFIG_PATH)

    return args

def get_clustering(fd,  clustering_type, vae : VAE = None, vae_sample=True) -> list:
    data = fd.data.drop(columns='label').values
    if vae:
        data = vae.encode(data, sample=vae_sample).detach().cpu().numpy()

    labels = fd.data['label'].unique()
    match clustering_type:
        case "none":
            return [sample['label'] for sample in fd.data.iloc]
        case "kmeans":
            print("Fitting kmeans...")
            kmeans = KMeans(len(labels), n_init='auto').fit(data)
            return kmeans.labels_
        case "dbscan":
            print("Fitting dbscan...")
            dbscan = DBSCAN(min_samples=len(labels)).fit(data)
            return dbscan.labels_

def get_clustering_plotting_config(clustering, plot_config, plotting_data):
    if len(clustering) == 0:
        return []
    
    print(f"{clustering[0]} {type(clustering[0])}c0")
    if type(clustering[0]) == str:
        return [(plotting_data[np.array(clustering) == n], plot_config[LABEL_CONFIG][n]) for n in np.unique(clustering)]
    
    return [(plotting_data[np.array(clustering) == i], {'label' : i, 'color' : '#000000', 'marker' : 'o'}) for i in np.unique(clustering)]

def load_plot_config(fd : FluxDataset, args):
    try:
        with open(args.plot_config_path, 'r') as plot_config_file:
            plot_config = json.load(plot_config_file)
    except:
        plot_config = {}
    
    def add(name, value, dic):
        if name not in dic: dic[name] = value

    add('figsize', (10, 8), plot_config)
    add('dot_size',1.0, plot_config)
    add('ldot_size', 30.0, plot_config)
    add('lfontsize', 8.0, plot_config)
    add('lbbox', None, plot_config)
    add('plot_width',1.0, plot_config)
    add(LABEL_CONFIG, {}, plot_config)

    for name in fd.data['label'].unique():
        label_config = plot_config[LABEL_CONFIG]
        add(name, {}, label_config)
        add('color', '#FF00FF', label_config[name])
        add('marker', 'o', label_config[name])
        add('label', name, label_config[name])

    with open(args.plot_config_path, 'w') as plot_config_file:
        json.dump(plot_config, plot_config_file, indent=4)
    return plot_config

def main():
    args = get_args()
    fd = FluxDataset(args.dataset, args.dataset_size, 0, args.dataset_join, True, args.dataset_reload_aux)
    plot_config = load_plot_config(fd, args)
    vae = None if args.vae_folder is None else make_VAE(args.vae_folder, args.vae_version) 

    plt.figure(figsize=plot_config['figsize'])
    ax = plt.subplot(111)

    data = fd.data.drop(columns='label').values
    clustering_data = data

    if vae is not None:
        data = vae.encode(data).detach().cpu().numpy()
        clustering_data = data if args.cluster_after_vae else clustering_data

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
    
    clustering = get_clustering(fd, args.clustering, vae if args.cluster_after_vae else None, args.vae_sample)
    clusters = get_clustering_plotting_config(clustering, plot_config, data) 

    for cluster in clusters:
        cluster_data, cluster_config  = cluster

        ax.scatter(cluster_data[:,0], cluster_data[:,1],
                    color=cluster_config['color'],
                    marker=cluster_config['marker'],
                    label=cluster_config['label'],
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