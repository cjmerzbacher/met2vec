import pandas as pd
import numpy as np
import torch
import os
import re
import random
import logging
import json

from cobra.io import read_sbml_model
from torch.utils.data import Dataset
from tqdm import tqdm

logging.getLogger('cobra').setLevel(logging.CRITICAL)

GEM_PATH_FOLDER = 'gems'
RENAME_DICT_FILE = "renaming.json"

def get_name_from_file(file : str):
    return re.sub('_[0-9|k]*.csv', '', re.search(r'[a-zA-Z \-_,]*_[0-9|k]+.csv', file).group())

def get_rename_dict(file : str):
    def get_reaction_name(reaction):
        return "".join(sorted([f'{m.name}({m.compartment})[{reaction.metabolites[m]}]' for m in reaction.metabolites]))

    try:
        model = read_sbml_model(file)
    except:
        return None
    return {r.id : get_reaction_name(r) for r in model.reactions}

class FluxDataset(Dataset):
    '''Class alowing a fluxdataset.csv file to be loaded into pytorch.'''
    def __init__(self, path, verbose=False, reload_renaming=False):
        '''Takes files - a path to a csv file containing the data to be leaded. The data is automatically normalized when loaded.'''
        self.path = path
        self.folder = os.path.dirname(path)
        self.verbose = verbose
        self.reload_renaming = reload_renaming

        self.files = [path] if path.endswith('.csv') else [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]
        self.files = {get_name_from_file(f) : f for f in self.files}

        self.find_renaming()
        self.find_columns()

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
            if name in self.renaming_dicts and not self.reload_renaming:
                continue
            gem_rename_dict = get_rename_dict(gem_file)
            if gem_rename_dict != None:
                self.renaming_dicts[name] = gem_rename_dict 

        with open(renaming_file_path, 'w') as file:
            json.dump(self.renaming_dicts, file)

    def read_csv(self, file, **kwargs):
        df = pd.read_csv(file, **kwargs)
        name = get_name_from_file(file)
        if name in self.renaming_dicts:
            df.rename(columns=self.renaming_dicts[name], inplace=True)
        df.fillna(0.0, inplace=True)
        return df

    def find_columns(self):
        columns = set()
        for file in self.files.values():
            columns = columns.union(self.read_csv(file, nrows=0, index_col=0).columns)
        self.columns = list(columns)

    def reload_mix(self):
        file = random.sample(sorted(self.files.values()), 1)[0]
        df = self.read_csv(file)
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


FluxDataset("./data/samples/small_human/", verbose=True)