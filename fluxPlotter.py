import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import distinctipy

from itertools import product
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from fluxDataset import FluxDataset
from vae import make_VAE, VAE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PLOT_CONFIG_PATH = '.plotconfig.json'
LABEL_CONFIG = 'lable_config'

PRE = 'pre'
EMB = 'emb'
REC = 'rec'
VAE_STAGES = [PRE, EMB, REC]

def boolean_string(s):
    if s.lower() not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string.')
    return s.lower() == 'true'

def get_args():
    parent_parser = argparse.ArgumentParser('Parent Parser', add_help=False)
    parent_parser.add_argument('dataset')
    parent_parser.add_argument('-n', '--dataset_size', type=int, default=1024)
    parent_parser.add_argument('-j', '--dataset_join', choices=['inner', 'outer'], default='inner')
    parent_parser.add_argument('-r', '--dataset_reload_aux', type=boolean_string, default=False)
    parent_parser.add_argument('--dataset_skip_tmp', default=False, type=boolean_string)
    parent_parser.add_argument('-v', '--vae_folder')
    parent_parser.add_argument('--vae_version', type=int)
    parent_parser.add_argument('--vae_sample', type=boolean_string, default=False)
    parent_parser.add_argument('-t', '--title')
    parent_parser.add_argument('-s', '--save_plot')

    dbscan_parser = argparse.ArgumentParser('DbScan Parameters', add_help=False)
    dbscan_parser.add_argument('--dbscan_eps', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    dbscan_parser.add_argument('--dbscan_mins', type=int, nargs=3, default=[5, 5, 5])

    comp_parser = argparse.ArgumentParser('Comp Parser', add_help=False)
    comp_parser.add_argument('--write_scores', type=boolean_string, default=False)

    # main parser
    parser = argparse.ArgumentParser('Flux Plotter')
    subparsers = parser.add_subparsers(dest='command')  

    # scatter
    scatter_parser = subparsers.add_parser('scatter', parents=[parent_parser, dbscan_parser]) 
    scatter_parser.add_argument('-p', '--preprocessing', choices=['none', 'tsne', 'pca'], default='none')
    scatter_parser.add_argument('--perplexity', type=float, default=30.0)
    scatter_parser.add_argument('--clustering', choices=['none', 'kmeans', 'dbscan'], default='none')
    scatter_parser.add_argument('--cluster_stage', choices=VAE_STAGES, default=EMB)
    scatter_parser.add_argument('--plot_stage', choices=VAE_STAGES, default=EMB)

    # ari
    ari_parser = subparsers.add_parser('ari', parents=[parent_parser, dbscan_parser, comp_parser])
    ari_parser.add_argument('-c', '--clusterings', nargs='+', default=['kmeans', 'dbscan'])
    ari_parser.add_argument('--vae_stages', nargs='+', default=VAE_STAGES)
    ari_parser.add_argument('--repeat', default=1, type=int)

    # gmm
    gmm_parser = subparsers.add_parser('gmm', parents=[parent_parser, comp_parser])
    gmm_parser.add_argument("--gmm_stage", choices=VAE_STAGES, default=EMB)
    gmm_parser.add_argument("--n_components", default=1, type=int)

    args = parser.parse_args()

    args.folder = os.path.dirname(args.dataset)
    args.plot_config_path = os.path.join(args.folder, PLOT_CONFIG_PATH)
    if hasattr(args, "dbscan_eps"):
        args.dbscan_params = {
            PRE  : (args.dbscan_eps[0], args.dbscan_mins[0]),
            EMB  : (args.dbscan_eps[1], args.dbscan_mins[1]),
            REC : (args.dbscan_eps[2], args.dbscan_mins[2])}

    return args


def get_data(fd : FluxDataset, stage : str, vae : VAE = None, vae_sample : bool = False, label : str = None):
    if label is None:
        data = fd.values
    else:
        data = fd[label]

    if vae and stage != PRE:
        data = vae.encode(data, sample=vae_sample)
        if stage == REC:
            data = vae.decode(data)
        data = data.detach().cpu().numpy()
    return data


def get_clustering(fd,  clustering_type, vae : VAE = None, vae_stage : str = EMB, vae_sample=True, dbscan_params=None) -> list:
    data = get_data(fd, vae_stage, vae, vae_sample)

    labels = fd.data['label'].unique()
    match clustering_type:
        case "none":
            return [sample['label'] for sample in fd.data.iloc]
        case "kmeans":
            kmeans = KMeans(len(labels), n_init='auto').fit(data)
            return kmeans.labels_
        case "dbscan":
            eps, mins = dbscan_params[vae_stage] if dbscan_params != None else (1.0, 5)
            dbscan = DBSCAN(eps=eps, min_samples=mins).fit(data)
            return dbscan.labels_

def apply_preprocessing(data, preprocessing : str, perplexity : float = 30):
    match preprocessing:
        case 'none':
            return data
        case 'tsne':
            tsne = TSNE(perplexity=perplexity)
            return tsne.fit_transform(data)
        case 'pca':
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
    add('dpi', 800, plot_config)
    add('dot_size',1.0, plot_config)
    add('ldot_size', 30.0, plot_config)
    add('lfontsize', 8.0, plot_config)
    add('lbbox', None, plot_config)
    add('plot_width',1.0, plot_config)
    add(LABEL_CONFIG, {}, plot_config)

    colors = [distinctipy.get_text_color(c) for c in distinctipy.get_colors(len(fd.data['label'].unique()))]

    for i, name in enumerate(fd.data['label'].unique()):
        label_config = plot_config[LABEL_CONFIG]
        add(name, {}, label_config)
        add('color', colors[i], label_config[name])
        add('marker', 'o', label_config[name])
        add('label', name, label_config[name])

    with open(args.plot_config_path, 'w') as plot_config_file:
        json.dump(plot_config, plot_config_file, indent=4)
    return plot_config

def get_clustering_set(fd, clustering_type, vae_stage, vae, args):
    clustering_set = []
    repeat = args.repeat if clustering_type != 'none' else 1
    with tqdm(range(repeat), desc=f'Repeating {clustering_type}-{vae_stage}', disable=repeat == 1, position=0) as t:
        for _ in t:
            clustering = get_clustering(fd, clustering_type, vae, vae_stage, args.vae_sample, args.dbscan_params)
            clustering_set.append(clustering)
                
            n_outliers = list(clustering).count(-1)
            n_clusters = len(set(clustering)) - (1 if n_outliers > 0 else 0)

            t.set_postfix({'clusters':n_clusters, 'outliers':n_outliers})
    return clustering_set

def plot_comparison(scores, names_x, names_y, std = None, write_scores = True, xlabel="", ylabel=""):
    ax = plt.subplot(111)
    ax.imshow(scores)

    ax.set_xticks(np.arange(scores.shape[0]), labels=names_x)
    ax.set_yticks(np.arange(scores.shape[1]), labels=names_y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if write_scores:
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                text =f'{scores[i,j]:.2f}'
                if std != None and std[i,j] > 0.005:
                    text += f'$\pm${std[i,j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', color='w')

    plt.tight_layout()

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

# Main Functions

def scatter_plot(args, fd, plot_config, vae):
    ax = plt.subplot(111)

    plotting_data = get_data(fd, args.plot_stage, vae, args.vae_sample)
    plotting_data = apply_preprocessing(plotting_data, args.preprocessing, args.perplexity)

    print(f"Fitting {args.clustering}...")
    clustering = get_clustering(fd, args.clustering, vae, args.cluster_stage, args.vae_sample, args.dbscan_params)
    clusters = get_clustering_plotting_config(clustering, plot_config, plotting_data) 

    n_clusters = len(clusters)
    n_outliers = list(clustering).count(-1)

    print(f"clusters -> {n_clusters} n_outliers -> {n_outliers}")

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


def ari_plot(args, fd, vae):
    options = [args.clusterings]
    options.append(args.vae_stages if vae else [PRE])

    clustering_sets = []
    names = []

    instances = list(product(*options)) + [('none', PRE)]

    for (clustering, vae_stage) in tqdm(instances, position=1):
        names.append(f"{clustering}-{vae_stage}")
        clustering_sets.append(get_clustering_set(fd, clustering, vae_stage, vae, args))
    
    ari_scores, ari_std = get_score_distribution(clustering_sets)
    plot_comparison(ari_scores, names, names, ari_std, args.write_scores, )


def gmm_plot(args, fd, vae):
    labels = list(set(fd.labels))
    gmms = {label : GaussianMixture(args.n_components) for label in labels}
    data_sets = {label : get_data(fd, args.gmm_stage, vae, args.vae_sample, label) for label in labels}

    for label, gmm in tqdm(gmms.items(), desc='Fitting models'):
        gmm.fit(data_sets[label])

    def pred(data):
        probs = np.array([gmms[label].score_samples(data).ravel() for label in labels])
        return np.argmax(probs, axis=0)

    def get_prediction_accuracy(exp_label, data_label):
        return np.mean(pred(data_sets[data_label]) == labels.index(exp_label))
    
    acc_scores, _ = get_score_distribution(labels, get_prediction_accuracy)
    plot_comparison(acc_scores, labels, labels, write_scores=args.write_scores, xlabel="True Label", ylabel="Prediction")



def main():
    args = get_args()
    fd = FluxDataset(args.dataset, args.dataset_size, 0, args.dataset_join, True, args.dataset_reload_aux, args.dataset_skip_tmp)
    vae = None if args.vae_folder is None else make_VAE(args.vae_folder, args.vae_version) 

    # set up figure
    plot_config = load_plot_config(fd, args)
    plt.figure(figsize=plot_config['figsize'])

    match args.command:
        case 'scatter':
            scatter_plot(args, fd, plot_config, vae)
        case 'ari':
            ari_plot(args, fd, vae)
        case 'gmm':
            gmm_plot(args, fd, vae)


    if args.save_plot is None:
        plt.show()
    else:
        plt.savefig(args.save_plot, dpi=plot_config['dpi'])

if __name__ == '__main__':
    main()