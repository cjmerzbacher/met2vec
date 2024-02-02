from .constants import *
from .fluxDataset import get_name_prefix
from random import randint

import argparse

def parser_fluxDataset_loading(name : str = "", path_tag=None):
    name, prefix = get_name_prefix(name)
    path_tag = [path_tag] if path_tag != None else []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*path_tag, f"--{prefix}path", required=True, help=f"The path {name}dataset will be laoded from.")
    parser.add_argument(f"--{prefix}size", default=65536, type=int, help=f"The size of {name}dataset samples.")
    parser.add_argument(f"--{prefix}model_folder", help=f"If set the model folder used for {name}dataset instead of main folder.")

    return parser

def parser_n(help : str = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", help=help, type=int, default=0)
    return parser

def boolean_string(s):
    if s.lower() not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string.')
    return s.lower() == 'true'

# Constant Parsers
PARSER_STAGE = argparse.ArgumentParser(add_help=False)
PARSER_STAGE.add_argument("-s", "--stage", choices=VAE_STAGES, help="The stage in the VAE used to evaluated data.")

PARSER_VAE_LOAD = argparse.ArgumentParser(add_help=False)
PARSER_VAE_LOAD.add_argument("-v", "--vae_folder", help="The folder the VAE will be loaded from.")
PARSER_VAE_LOAD.add_argument("--vae_version", help="The version of the VAE that will be loaded.")
PARSER_VAE_LOAD.add_argument("--legacy_vae", type=boolean_string, help="If the VAE used legacy sigma encoding.", default=False)

PARSER_PREP = argparse.ArgumentParser(add_help=False)
PARSER_PREP.add_argument("--prep", default=NONE, choices=PREPS, help='The preprocessing that will be used on the data.')
PARSER_PREP.add_argument("--perp", type=float, default=30, help="The perplexit for TSNE (if used).")

PARSER_SAVE = argparse.ArgumentParser(add_help=False)
PARSER_SAVE.add_argument("--save_path", help="Where output will be saved.", required=True)

PARSER_VAE_SAMPLE = argparse.ArgumentParser(add_help=False)
PARSER_VAE_SAMPLE.add_argument("--sample", default=True, type=boolean_string, help="If the VAE should be used in sample mode.")

PARSER_KMEANS_K = argparse.ArgumentParser(add_help=False)
PARSER_KMEANS_K.add_argument("-k", type=int, help="The number of cluster, if unset same as number of labels.")

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