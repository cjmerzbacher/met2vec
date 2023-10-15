import pandas as pd
import numpy as np
import torch
import os
import re
import random
import logging
import json
import pickle

from cobra.io import read_sbml_model
from torch.utils.data import Dataset
from tqdm import tqdm

logging.getLogger('cobra').setLevel(logging.CRITICAL)

GEM_PATH_FOLDER = 'gems'
RENAME_DICT_FILE = ".renaming.json"
JOIN_FILE = ".join.json"
PKL_FOLDER = ".pkl"
DEFAULT_DATASET_SIZE = 65536

def get_name_from_file(file : str):
    return re.sub('_[0-9|k]*.csv', '', re.search(r'[a-zA-Z \-_,]*_[0-9|k]+.csv', file).group())

def get_rename_dict(file : str):
    def get_reaction_name(reaction):
        reaction_parts = [f'{m.name}({m.compartment})[{reaction.metabolites[m]}]' for m in reaction.metabolites]
        name = "".join(sorted(reaction_parts))
        return name.replace(",", ".")

    try:
        model = read_sbml_model(file)
    except:
        return None
    return {r.id : get_reaction_name(r) for r in model.reactions}

def get_non_zero_columns(df : pd.DataFrame):
    non_zeros = np.any(df.values != 0.0, axis=0)
    return df.columns[non_zeros]
class FluxDataset(Dataset):
    '''Class alowing a fluxdataset.csv file to be loaded into pytorch.'''
    def __init__(self, path, dataset_size=DEFAULT_DATASET_SIZE, join='inter', verbose=False, reload_aux=False):
        '''Takes files - a path to a csv file containing the data to be leaded. The data is automatically normalized when loaded.'''
        self.path = path
        self.dataset_size = dataset_size
        self.join = join
        self.folder = os.path.dirname(path)
        self.pkl_folder = os.path.join(self.folder, PKL_FOLDER)
        self.verbose = verbose

        self.reload = reload_aux

        self.files = [path] if path.endswith('.csv') else [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]
        self.files = {get_name_from_file(f) : f for f in self.files}

        self.find_renaming()
        self.find_joins()

        self.reload_mix()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])
    
    def find_renaming(self):
        self.renaming_dicts = {}
        if len(self.files) < 2:
            return

        try:
            renaming_file_path = os.path.join(self.folder, RENAME_DICT_FILE)
            with open(renaming_file_path, 'r') as file:
                self.renaming_dicts = json.load(file)
        except:
            pass

        for name, file in tqdm(self.files.items(), desc='Loading Renaming', disable=not self.verbose):
            gem_file = os.path.join(self.folder, GEM_PATH_FOLDER, f"{name}.xml")
            if name in self.renaming_dicts and not self.reload:
                continue
            gem_rename_dict = get_rename_dict(gem_file)
            if gem_rename_dict != None:
                self.renaming_dicts[name] = gem_rename_dict 

        with open(renaming_file_path, 'w') as file:
            json.dump(self.renaming_dicts, file)

    def get_pkl_path(self, name : str):
        return os.path.join(self.pkl_folder, f"{name}.pkl")

    def get_df(self, name : str) -> pd.DataFrame:
        pkl_path = self.get_pkl_path(name)
        try:
            with open(pkl_path, 'rb') as pkl_file:
                df = pickle.load(pkl_file)
        except:
            df = pd.read_csv(self.files[name], index_col=0, engine='pyarrow')
            if not os.path.exists(self.pkl_folder):
                os.makedirs(self.pkl_folder)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(df, pkl_file)

        return df.rename(columns=self.renaming_dicts[name])

    def find_joins(self):
        join_path = os.path.join(self.folder, JOIN_FILE)
        if os.path.exists(join_path) and not self.reload:
            with open(join_path, 'r') as join_file:
                self.joins = json.load(join_file)
        else:
            inner = set()
            outer = set()

            for i, name in tqdm(enumerate(self.files), desc="Making inter_union", disable=not self.verbose):
                columns = get_non_zero_columns(self.get_df(name))
                inner = set(columns) if i == 0 else inner.intersection(columns)
                outer = outer.union(columns)

            self.joins = {
                'inner' : list(inner),
                'outer' : list(outer)
            }

            with open(join_path, 'w') as join_file:
                json.dump(self.joins, join_file, indent=4)

    def reload_mix(self):
        df = pd.DataFrame(columns=self.joins[self.join])
        self.labels = []
        for name in tqdm(self.files, desc='Loading mix...', disable=not self.verbose):
            sample_df = self.get_df(name)
            n_sample = min(self.dataset_size // len(self.files), len(sample_df))

            sample_df = sample_df.sample(n_sample)
            df = pd.concat([df, sample_df], join=self.join, ignore_index=True)
            self.labels += [name] * n_sample

        self.data = np.array(df.values, dtype=float)
        self.normalize()

    def normalize(self):
        '''Normalized the loaded data for allow columns which are not all 0. The resulting mean and std are stored in the class.'''
        ignore = np.repeat(np.all(self.data == 0.0, axis=0, keepdims=True), self.data.shape[0], axis=0)

        self.mean = np.mean(self.data, axis=0)
        self.data = (self.data - self.mean)

        self.std = np.std(self.data, axis=0, ddof=1.0)
        self.std[np.all(ignore, axis=0)] = 1.0

        self.data = self.data / self.std

        self.data[ignore] = 0.0
