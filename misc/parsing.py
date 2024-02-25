from .constants import *
from .fluxDataset import get_name_prefix
from random import randint

import argparse

def print_args(args : argparse.Namespace):
    for name, val in vars(args).items():
        print(f"    {name}: {val}")

def parser_fluxDataset_loading(name : str = "", path_tag=None):
    name, prefix = get_name_prefix(name)
    path_tag = [path_tag] if path_tag != None else []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*path_tag, f"--{prefix}path", required=True, help=f"The path {name}dataset will be laoded from.")
    parser.add_argument(f"--{prefix}size", default=65536, type=int, help=f"The size of {name}dataset samples.")
    parser.add_argument(f"--{prefix}model_folder", help=f"If set the model folder used for {name}dataset instead of main folder.")

    return parser

def parser_multiple_fluxDatasets_loading(name : str = ""):
    name, prefix = get_name_prefix(name)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(f"--{prefix}paths", required=True, nargs="+", help=f"The paths {name}datasets will be laoded from.")
    parser.add_argument(f"--{prefix}size", default=65536, type=int, help=f"The size of {name}datasets' samples.")
    parser.add_argument(f"--{prefix}model_folder", help=f"If set the model folder used for {name}datasets instead of main folder.")

    return parser

def parser_n(help : str = None, default=0):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", help=help, type=int, default=default)
    return parser

def boolean_string(s):
    if s.lower() not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string.')
    return s.lower() == 'true'

# Constant Parsers
PARSER_STAGE = argparse.ArgumentParser(add_help=False)
PARSER_STAGE.add_argument("-s", "--stage", default=PRE, choices=VAE_STAGES, help="The stage in the VAE used to evaluated data.")

PARSER_VAE_LOAD = argparse.ArgumentParser(add_help=False)
PARSER_VAE_LOAD.add_argument("-v", "--vae_folder", help="The folder the VAE will be loaded from.")
PARSER_VAE_LOAD.add_argument("--vae_version", help="The version of the VAE that will be loaded.")
PARSER_VAE_LOAD.add_argument("--legacy_vae", type=boolean_string, help="If the VAE used legacy sigma encoding.", default=False)

PARSER_VAE_FOLDERS = argparse.ArgumentParser(add_help=False)
PARSER_VAE_FOLDERS.add_argument("--vae_folders", type=str, nargs='+', help="The folders from which VAEs will be loaded.")

PARSER_PREP = argparse.ArgumentParser(add_help=False)
PARSER_PREP.add_argument("--prep", default=NONE, choices=PREPS, help='The preprocessing that will be used on the data.')
PARSER_PREP.add_argument("--perp", type=float, default=30, help="The perplexit for TSNE (if used).")

PARSER_SAVE = argparse.ArgumentParser(add_help=False)
PARSER_SAVE.add_argument("--save_path", help="Where output will be saved.", required=True)

PARSER_VAE_SAMPLE = argparse.ArgumentParser(add_help=False)
PARSER_VAE_SAMPLE.add_argument("--sample", default=True, type=boolean_string, help="If the VAE should be used in sample mode.")

PARSER_KMEANS_K = argparse.ArgumentParser(add_help=False)
PARSER_KMEANS_K.add_argument("-k", type=int, help="The number of cluster, if unset same as number of labels.")

PARSER_KMEANS_METRIC = argparse.ArgumentParser(add_help=False)
PARSER_KMEANS_METRIC.add_argument("--kmeans_metric", choices=KMEANS_METRICS)

PARSER_MAX_K = argparse.ArgumentParser(add_help=False)
PARSER_MAX_K.add_argument("--max_k", type=int, help="The max k that will be used, if unset nr files will be used.")

PARSER_MODEL_FOLDER = argparse.ArgumentParser(add_help=False)
PARSER_MODEL_FOLDER.add_argument("--model_folder", help="If specified, common folder all datasets will used for model files.")

PARSER_FLUXES = argparse.ArgumentParser(add_help=False)
PARSER_FLUXES.add_argument("--fluxes", help="The names of fluxes to be extracted.", nargs='*', default=[])

PARSER_SEED = argparse.ArgumentParser(add_help=False)
PARSER_SEED.add_argument("--seed", type=int, default=randint(0,65536))

PARSER_VERBOSE = argparse.ArgumentParser(add_help=False)
PARSER_VERBOSE.add_argument("--verbose", action="store_true", default=False)

PARSER_JOIN = argparse.ArgumentParser(add_help=False)
PARSER_JOIN.add_argument("--join", choices=[INNER, OUTER], default=INNER, help="The join that wll be used on the dataset.")

PARSER_GROUP_BY = argparse.ArgumentParser(add_help=False)
PARSER_GROUP_BY.add_argument("--group_by", choices=SOURCE_COLUMNS, default=LABEL, help="How the data will be grouped.")

PARSER_STAGES = argparse.ArgumentParser(add_help=False)
PARSER_STAGES.add_argument("--stages", choices=VAE_STAGES, nargs="+", default=VAE_STAGES, help="VAE stages that will be used.")

PARSER_ORIGIONAL_CLUSTERING = argparse.ArgumentParser(add_help=False)
PARSER_ORIGIONAL_CLUSTERING.add_argument("--origional_clustering", choices=SOURCE_COLUMNS, default=LABEL, help="The data source columns that will be used as the origional labels.")

PARSER_BOOTSTRAP_N = argparse.ArgumentParser(add_help=False)
PARSER_BOOTSTRAP_N.add_argument("--bootstrap_n", type=int, default=128, help="Number of bootstrap repititions that will be made to calculate mean and variance for ari.")

PARSER_BETA_S = argparse.ArgumentParser(add_help=False)
PARSER_BETA_S.add_argument("--beta_S", type=float, default=0.0, help="Weighting value for the stoicheometry loss.")

PARSER_IGNORE_CHEMICAL_NAME = argparse.ArgumentParser(add_help=False)
PARSER_IGNORE_CHEMICAL_NAME.add_argument("--ignore_chemical_name", action="store_true", help="Whether the chemical_name in the reaction formula should be ignored.")




