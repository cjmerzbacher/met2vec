from .constants import *
from .fluxDataset import get_name_prefix

import argparse

def fluxDataset_loading_parser(name : str = "", path_tag=None):
    name, prefix = get_name_prefix(name)
    path_tag = [path_tag] if path_tag != None else []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*path_tag, f"--{prefix}path", required=True, help=f"The path {name}dataset will be laoded from.")
    parser.add_argument(f"--{prefix}size", default=65536, type=int, help=f"The size of {name}dataset samples.")
    parser.add_argument(f"--{prefix}model_folder", help=f"If set the model folder used for {name}dataset instead of main folder.")
    parser.add_argument(f"--{prefix}join", default=INNER, choices=DATASET_JOINS, help=f"The join that will be used for {name}dataset.")
    parser.add_argument(f"--{prefix}verbose", default=True, type=boolean_string, help="Whether the dataset will print debug messages")
    parser.add_argument(f"--{prefix}reload_aux", type=boolean_string, default=False, help=f"Whether {name}dataset should reload aux.")
    parser.add_argument(f"--{prefix}skip_tmp", type=boolean_string, default=False, help=f"Whether {name}dataset should reload the tmp files.")

    return parser

def boolean_string(s):
    if s.lower() not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string.')
    return s.lower() == 'true'

# Constant Parsers
PARSER_STAGE = argparse.ArgumentParser(add_help=False)
PARSER_STAGE.add_argument("-s", "--stage", choices=VAE_STAGES, help="The stage in the VAE used to evaluated data.")

PARSER_LOAD_VAE = argparse.ArgumentParser(add_help=False)
PARSER_LOAD_VAE.add_argument("-v", "--vae_folder", help="The folder the VAE will be loaded from.")
PARSER_LOAD_VAE.add_argument("--vae_version", help="The version of the VAE that will be loaded.")

PARSER_PREP = argparse.ArgumentParser(add_help=False)
PARSER_PREP.add_argument("--prep", default=NONE, choices=PREPS, help='The preprocessing that will be used on the data.')
PARSER_PREP.add_argument("--perp", type=float, default=30, help="The perplexit for TSNE (if used).")

PARSER_SAVE = argparse.ArgumentParser(add_help=False)
PARSER_SAVE.add_argument("--save_path", help="Where output will be saved.", required=True)

PARSER_SAMPLE = argparse.ArgumentParser(add_help=False)
PARSER_SAMPLE.add_argument("--sample", default=True, type=boolean_string, help="If the VAE should be used in sample mode.")

PARSER_KMEANS_K = argparse.ArgumentParser(add_help=False)
PARSER_KMEANS_K.add_argument("-k", type=int, help="The number of cluster, if unset same as number of labels.")

PARSER_MODEL_FOLDER = argparse.ArgumentParser(add_help=False)
PARSER_MODEL_FOLDER.add_argument("--model_folder", help="If specified, common folder all datasets will used for model files.")