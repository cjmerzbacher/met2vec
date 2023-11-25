from .constants import *
from .fluxDataset import get_name_prefix

import argparse

def fluxDataset_loading_parser(name : str = "", path_tag=None):
    name, prefix = get_name_prefix(name)
    path_tag = [path_tag] if path_tag != None else []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*path_tag, f"--{prefix}path", required=True, help=f"The path {name}dataset will be laoded from.")
    parser.add_argument(f"--{prefix}size", default=65536, type=int, help=f"The size of {name}dataset samples.")
    parser.add_argument(f"--{prefix}test_size", default=0, type=int, help=f"The size of the test set for {name}dataset.")
    parser.add_argument(f"--{prefix}join", default=INNER, choices=DATASET_JOINS, help=f"The join that will be used for {name}dataset.")
    parser.add_argument(f"--{prefix}verbose", default=True, type=boolean_string, help="Whether the dataset will print debug messages")
    parser.add_argument(f"--{prefix}reload_aux", type=boolean_string, default=False, help=f"Whether {name}dataset should reload aux.")
    parser.add_argument(f"--{prefix}skip_tmp", type=boolean_string, default=False, help=f"Whether {name}dataset should reload the tmp files.")

    return parser

def get_save_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save_path", help="Where output will be saved.")

    return parser

def get_title_parser(name: str = None, default : str = None):
    name = f"{name}_" if name != None else ""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(f"--{name}title", default=default)

    return parser

def boolean_string(s):
    if s.lower() not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string.')
    return s.lower() == 'true'