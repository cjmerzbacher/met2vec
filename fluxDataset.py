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
DEFAULT_TEST_SIZE = 2048

rm = os.unlink
joinp = os.path.join

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
    def __init__(self, path, dataset_size=DEFAULT_DATASET_SIZE, test_size=DEFAULT_TEST_SIZE, join='inner', verbose=False, reload_aux=False):
        '''Takes files - a path to a csv file containing the data to be leaded. The data is automatically normalized when loaded.'''
        self.set_folder(path)
        self.join = join
        self.verbose = verbose
        self.reload = reload_aux

        # Find renamings and joins
        self.find_renaming()
        self.find_joins()

        # Load data into current
        if not self.make_tmp_files(dataset_size, test_size):
            print("Loading FluxDataset failed")
            return
        self.load_sample()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.labels[idx], torch.Tensor(self.values[idx])
    
    def set_folder(self, path : str):
        self.path = path
        self.folder = os.path.dirname(path)
        self.pkl_folder = os.path.join(self.folder, PKL_FOLDER)
        self.test_pkl_folder = os.path.join(self.pkl_folder, 'test')
        self.train_pkl_folder = os.path.join(self.pkl_folder, 'train')

        self.files = [path] if path.endswith('.csv') else [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]
        self.files = {get_name_from_file(f) : f for f in self.files}

    def find_renaming(self):
        self.renaming_dicts = {}
        if len(self.files) < 2:
            return

        try:
            renaming_file_path = os.path.join(self.folder, RENAME_DICT_FILE)
            with open(renaming_file_path, 'r') as file:
                self.renaming_dicts = json.load(file)
        except:
            self.renaming_dicts = {}

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

    def make_tmp_files(self, dataset_size : int, test_size : int):
        samples_per_file = dataset_size // len(self.files)
        test_per_file = test_size // len(self.files)

        ensure_exists = lambda f: None if os.path.exists(f) else os.makedirs(f)

        ensure_exists(self.test_pkl_folder)
        ensure_exists(self.train_pkl_folder)

        if not os.path.exists(self.test_pkl_folder):
            os.makedirs(self.test_pkl_folder)

        for name in tqdm(self.files, desc='Making tmps', disable=not self.verbose):
            df = self.get_df(name)
            if len(df.index) < samples_per_file + test_per_file:
                print(f'Unable to make tmp files! Sample "{name}" ({len(df.index)}) to small: spf {samples_per_file}, ts {test_size}.')
                return False

            [rm(joinp(self.test_pkl_folder,  f)) for f in os.listdir(self.test_pkl_folder)  if name in f]
            [rm(joinp(self.train_pkl_folder, f)) for f in os.listdir(self.train_pkl_folder) if name in f]

            def sample_drop(n):
                n = min(n, len(df.index))
                sample = df.sample(n)
                df.drop(sample.index, inplace=True)
                return sample
            
            def save_pkl(path, obj):
                with open(path, 'wb') as file:
                    pickle.dump(obj, file)
        
            train_df = sample_drop(test_per_file)
            save_pkl(joinp(self.train_pkl_folder, f'{name}.pkl'), train_df)

            n_saved = 0 
            while len(df.index) != 0:
                sample_df = sample_drop(samples_per_file)
                save_pkl(joinp(self.test_pkl_folder, f'{name}_{n_saved}.pkl'), sample_df)
                n_saved += 1

        return True

    def get_single_sample(self, name : str, is_test=False):
        folder = self.test_pkl_folder if is_test else self.train_pkl_folder
        paths = [joinp(folder, f) for f in os.listdir(folder) if name in f]
        path = random.sample(paths, 1)[0]

        with open(path, 'rb') as file:
            return pickle.load(file)
            

    def load_sample(self, is_test=False):
        df = pd.DataFrame(columns=self.joins[self.join])
        labels = []
        for name in tqdm(self.files, desc='Loading sample', disable=not self.verbose):
            tmp_sample_df = self.get_single_sample(name, is_test)
            labels += [name] * len(tmp_sample_df.index)
            df = pd.concat([df, tmp_sample_df], join=self.join, ignore_index=True)

        df = (df-df.mean(numeric_only=True))/df.std(numeric_only=True)
        df.fillna(0, inplace=True)

        self.values = df.values
        self.data = pd.concat([df, pd.DataFrame({'label' : labels})], axis=1)
        self.labels = labels

