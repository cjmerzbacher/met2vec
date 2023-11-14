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
from constants import *
from vae import VAE

logging.getLogger('cobra').setLevel(logging.CRITICAL)

GEM_PATH_FOLDER = 'gems'
RENAME_DICT_FILE = ".renaming.json"
JOIN_FILE = ".join.json"
PKL_FOLDER = ".pkl"
DEFAULT_DATASET_SIZE = 65536
DEFAULT_TEST_SIZE = 2048

rm = os.unlink
joinp = os.path.join
ensure_exists = lambda f: None if os.path.exists(f) else os.makedirs(f)

def get_name_from_file(file : str):
    return re.sub('_[0-9|k]*.csv', '', re.search(r'[a-zA-Z \-_,()0-9]*_[0-9|k]+.csv', file).group())

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

def make_tmp(path : str, n : int, source_df : pd.DataFrame):
    """Splits of a tmp file from source df of size n to be stored in path."""
    sample = source_df.sample(min(n, len(source_df.index)))
    source_df.drop(index=sample.index, inplace=True)
    with open(path, 'wb') as file: pickle.dump(sample, file)




class FluxDataset(Dataset):
    '''Class alowing a fluxdataset.csv file to be loaded into pytorch.'''
    def __init__(self, 
                 path : str, 
                 dataset_size : int = DEFAULT_DATASET_SIZE, 
                 test_size : int = DEFAULT_TEST_SIZE, 
                 join : str ='inner', 
                 verbose : bool = False, 
                 reload_aux : bool = False, 
                 skip_tmp : bool = False):
        '''Takes files - a path to a csv file containing the data to be leaded. 
        
        The data is automatically normalized when loaded.
        
        Args:
            path: The path (.csv file or folder containing .csv files).
            dataset_size: The size'''
        self.set_folder(path)
        self.join = join
        self.verbose = verbose
        self.reload = reload_aux

        # Find renamings and joins
        self.find_renaming()
        self.find_joins(self.files)

        # Load data into current
        if not skip_tmp:
            self.create_tmp_archive(dataset_size, test_size)
        self.load_sample()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if type(idx) == str:
            return self.values[np.array(self.labels) == idx]
        return self.labels[idx], torch.Tensor(self.values[idx])
    
    def set_folder(self, path : str):
        self.path = path
        self.folder = os.path.dirname(path)
        self.pkl_folder = os.path.join(self.folder, PKL_FOLDER)
        self.test_pkl_folder = os.path.join(self.pkl_folder, 'test')
        self.train_pkl_folder = os.path.join(self.pkl_folder, 'train')
        self.join_path = os.path.join(self.folder, JOIN_FILE)

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
            if self.verbose: print("No renaming dict found...")
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

        if name in self.renaming_dicts:
            return df.rename(columns=self.renaming_dicts[name])
        else:
            return df
    
    def find_joins(self, files : list[str]):
        """Finds the different joints for the dataset.
        
        Args:
            files: The files which the join will be defined over."""
        if os.path.exists(self.join_path) and not self.reload:
            with open(self.join_path, 'r') as join_file:
                self.joins = json.load(join_file)
            return

        inner, outer = set(), set()

        for i, name in tqdm(enumerate(files), desc="Making inter_union", disable=not self.verbose):
            columns = get_non_zero_columns(self.get_df(name))
            inner = set(columns) if i == 0 else inner.intersection(columns)
            outer = outer.union(columns)

        self.joins = {
            'inner' : list(inner),
            'outer' : list(outer)
        }

        with open(self.join_path, 'w') as join_file:
            json.dump(self.joins, join_file, indent=4)

    def create_tmp_archive(self, train_size : int, test_size : int):
        """Makes tmp files to be used to speed up sample loading.
        
        tmp files - These are pickled pd.DataFrame random subsets of the files loaded. 
        
        Args:
            train_size: The size of the train sample.
            test_size: The size of the test sample.

        Raises:
            ValueError: If train_size + test_size is greater than the number of samples
            for some sample we cannot split the data properly and an error will be thrown
        """
        train_per_file = train_size // len(self.files)
        test_per_file = test_size // len(self.files)

        ensure_exists(self.test_pkl_folder)
        ensure_exists(self.train_pkl_folder)

        for name in tqdm(self.files, desc='Making tmps', disable=not self.verbose):
            self.make_tmp_files(name, train_per_file, test_per_file)

    def remove_tmp_files(self, name):
        "Removes tmp files for a certain file name."
        [rm(joinp(self.test_pkl_folder,  f)) for f in os.listdir(self.test_pkl_folder)  if name in f]
        [rm(joinp(self.train_pkl_folder, f)) for f in os.listdir(self.train_pkl_folder) if name in f]

    def make_tmp_files(self, name : str, train : int, test : int):
        """Makes tmp files for a specific csv file."""
        df = self.get_df(name)
        required_samples = train + test

        if len(df.index) < required_samples:
            raise ValueError(
                f'Unable to make tmp files!' + 
                f'Sample "{name}" ({len(df.index)}) to small: spf {train}, ts {test}.'
                )

        self.remove_tmp_files(name)
        make_tmp(joinp(self.test_pkl_folder, f"{name}.pkl"), test, df) 

        n_saved = 0 
        while len(df.index) > 0.8 * train:
            make_tmp(joinp(self.train_pkl_folder, f"{name}_{n_saved}.pkl"), train, df)
            n_saved += 1

    def load_tmp_file(self, name : str, is_test=False):
        """Loads a tmp file for a given name.
        
        Args:
            name: The name of the for which the tmp will be loaded.
            is_test: If true the test tmp file will be loaded.

        Raises:
            FileNotFoundError: If there is no tmp file for name.
        """
        folder = self.test_pkl_folder if is_test else self.train_pkl_folder
        paths = [joinp(folder, f) for f in os.listdir(folder) if name in f]

        if len(paths) == 0:
            raise FileNotFoundError(f"No tmp files found for {name}.")

        path = random.sample(paths, 1)[0]
        with open(path, 'rb') as file:
            return pickle.load(file)

    def load_dataFrame(self, df : pd.DataFrame, labels : list[str]) -> None:
        """Loads in and normalizes a dataFrame."""      
        df = (df-df.mean(numeric_only=True))/df.std(numeric_only=True)
        df.fillna(0, inplace=True)
        
        self.values = df.values
        self.data = pd.concat([df, pd.DataFrame({'label' : labels})], axis=1)
        self.labels = labels

    def load_sample(self, is_test=False) -> None:
        """Loads a sample into the dataset.
        
        Args:
            is_test: If tre the sample loaded will be the test sample.
        """

        df = pd.DataFrame(columns=self.joins[self.join])
        labels = []
        for name in tqdm(self.files, desc='Loading sample', disable=not self.verbose):
            tmp_sample_df = self.load_tmp_file(name, is_test)
            labels += [name] * len(tmp_sample_df.index)
            df = pd.concat([df, tmp_sample_df], join=self.join, ignore_index=True)

        self.load_dataFrame(df, labels)


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
