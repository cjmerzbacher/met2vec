import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import distinctipy

from itertools import product
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from fluxDataset import FluxDataset
from vae import make_VAE, VAE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PLOT_CONFIG_PATH = '.plotconfig.json'
LABEL_CONFIG = 'lable_config'


def get_args():
    parent_parser = argparse.ArgumentParser('Parent Parser', add_help=False)
    parent_parser.add_argument('dataset')
    parent_parser.add_argument('-n', '--dataset_size', type=int, default=1024)
    parent_parser.add_argument('-j', '--dataset_join', choices=['inner', 'outer'], default='inner')
    parent_parser.add_argument('-r', '--dataset_reload_aux', type=bool, default=False)
    parent_parser.add_argument('--dataset_skip_tmp', default=False, type=bool)
    parent_parser.add_argument('-v', '--vae_folder')
    parent_parser.add_argument('--vae_version', type=int)
    parent_parser.add_argument('--vae_sample', type=bool, default=False)
    parent_parser.add_argument('-t', '--title')
    parent_parser.add_argument('-s', '--save_plot')
    parent_parser.add_argument('--dbscan_eps', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parent_parser.add_argument('--dbscan_mins', type=float, nargs=3, default=[5, 5, 5])

    # main parser
    parser = argparse.ArgumentParser('Flux Plotter')
    subparsers = parser.add_subparsers(dest='command')  

    # scatter
    scatter_parser = subparsers.add_parser('scatter', parents=[parent_parser]) 
    scatter_parser.add_argument('-p', '--preprocessing', choices=['none', 'tsne', 'pca'], default='none')
    scatter_parser.add_argument('--perplexity', type=float, default=30.0)
    scatter_parser.add_argument('--clustering', choices=['none', 'kmeans', 'dbscan'], default='none')
    scatter_parser.add_argument('--vae_stage', choices=['pre', 'emb', 'post'], default='emb')

    # ari
    ari_parser = subparsers.add_parser('ari', parents=[parent_parser])
    ari_parser.add_argument('-c', '--clusterings', nargs='+', default=['kmeans', 'dbscan'])
    ari_parser.add_argument('--vae_stages', nargs='+', default=['pre', 'emb', 'post'])

    args = parser.parse_args()
    args.folder = os.path.dirname(args.dataset)
    args.plot_config_path = os.path.join(args.folder, PLOT_CONFIG_PATH)
    args.dbscan_params = {
        'pre'  : (args.dbscan_eps[0], args.dbscan_mins[0]),
        'emb'  : (args.dbscan_eps[0], args.dbscan_mins[0]),
        'post' : (args.dbscan_eps[0], args.dbscan_mins[0])}

    return args


def get_clustering(fd,  clustering_type, vae : VAE = None, vae_stage : str = 'emb', vae_sample=True, dbscan_params=None) -> list:

    # Get the correct data to cluster on
    data = fd.data.drop(columns='label').values
    if vae and vae_stage != 'pre':
        data = vae.encode(data, sample=vae_sample)
        if vae_stage == 'post':
            data = vae.decode(data)
        data = data.detach().cpu().numpy()

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
            eps, mins = dbscan_params[vae_stage] if dbscan_params != None else (1.0, 5)
            dbscan = DBSCAN(eps=eps, min_samples=mins).fit(data)
            return dbscan.labels_

def scatter_preprocessing(data, args):
    match args.preprocessing:
        case 'none':
            return data
        case 'tsne':
            print('Fitting tsne...')
            tsne = TSNE(perplexity=args.perplexity)
            return tsne.fit_transform(data)
        case 'pca':
            print("Fitting PCA...")
            pca = PCA()
            return pca.fit_transform(data)

def get_clustering_plotting_config(clustering, plot_config, plotting_data):
    if len(clustering) == 0:
        return []
    
    if type(clustering[0]) == str:
        return [(plotting_data[np.array(clustering) == n], plot_config[LABEL_CONFIG][n]) for n in np.unique(clustering)]

    colors = distinctipy.get_colors(len(np.unique(clustering)))
    return [(plotting_data[np.array(clustering) == i], {'label' : i, 'color' : colors[i], 'marker' : 'o'}) for i in np.unique(clustering)]


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


def scatter_plot(args, fd, plot_config, vae, ax):
    data = fd.data.drop(columns='label').values

    if vae is not None:
        data = vae.encode(data).detach().cpu().numpy()

    plotting_data = scatter_preprocessing(data, args)
    clustering = get_clustering(fd, args.clustering, vae, args.vae_stage, args.vae_sample, args.dbscan_params)
    clusters = get_clustering_plotting_config(clustering, plot_config, plotting_data) 

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

def ari_plot(args, fd, vae, ax : plt.Axes):
    options = []
    options.append(args.clusterings)
    options.append(args.vae_stages if vae else ['pre'])

    clusterings = []
    names = []
    for instance in product(*options):
        clustering = get_clustering(fd, clustering_type=instance[0], vae=vae, vae_stage=instance[1], vae_sample=args.vae_sample, dbscan_params=args.dbscan_params)
        clusterings.append(clustering)
        print(clustering)
        names.append('-'.join(instance))
    
    clusterings.append(get_clustering(fd, 'none'))
    names.append('labels')
    n_clusterings = len(clusterings)

    ari_scores = np.zeros((n_clusterings, n_clusterings))
    for i in range(len(clusterings)):
        for j in range(len(clusterings)):
            ari_scores[i,j] = adjusted_rand_score(clusterings[i], clusterings[j])

    ax.imshow(ari_scores)
    ax.set_xticks(np.arange(n_clusterings), labels=names)
    ax.set_yticks(np.arange(n_clusterings), labels=names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(len(clusterings)):
        for j in range(len(clusterings)):
            ax.text(j, i, f'{ari_scores[i,j]:.2f}', ha='center', va='center', color='w')

    plt.tight_layout()




def main():
    args = get_args()
    fd = FluxDataset(args.dataset, args.dataset_size, 0, args.dataset_join, True, args.dataset_reload_aux, args.dataset_skip_tmp)
    plot_config = load_plot_config(fd, args)
    vae = None if args.vae_folder is None else make_VAE(args.vae_folder, args.vae_version) 

    plt.figure(figsize=plot_config['figsize'])
    ax = plt.subplot(111)

    match args.command:
        case 'scatter':
            scatter_plot(args, fd, plot_config, vae, ax)
        case 'ari':
            ari_plot(args, fd, vae, ax)


    if args.save_plot is None:
        plt.show()
    else:
        plt.savefig(args.save_plot, dpi=plot_config['dpi'])

 

if __name__ == '__main__':
    main()