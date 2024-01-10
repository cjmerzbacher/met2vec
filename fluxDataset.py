import pandas as pd
import numpy as np
import torch
import os
import re
import random
import logging
import json
import pickle

from numpy.random import RandomState
from cobra.io import read_sbml_model
from torch.utils.data import Dataset
from tqdm import tqdm
from misc.constants import *
from vae import VAE

logging.getLogger('cobra').setLevel(logging.CRITICAL)

GEM_PATH_FOLDER = 'gems'
RENAME_DICT_FILE = ".renaming.json"
JOIN_FILE = ".join.json"
PKL_FOLDER = ".pkl"
DEFAULT_DATASET_SIZE = 65536

rm = os.unlink
joinp = os.path.join
ensure_exists = lambda f: None if os.path.exists(f) else os.makedirs(f)

def get_file_name_from_sample_file(file : str):
    end_pattern = '.csv'
    return re.sub(end_pattern, '', re.search(f'[a-zA-Z \-_,()0-9]*_[0-9|k]*[\((0-9)*\)]*{end_pattern}', file).group())

def get_model_name_from_file_name(file : str):
    """Extracts the common name between sbml model and the sample file."""
    end_pattern = '_[0-9|k]+[\((0-9)*\)]*'
    return re.sub(end_pattern, '', file)

def get_model_name_from_sample_file(file : str):
    file_name = get_file_name_from_sample_file(file)
    model_name = get_model_name_from_file_name(file_name)
    return model_name

def get_gem_file(model_name : str, main_folder : str):
    return os.path.join(main_folder, GEM_PATH_FOLDER, f"{model_name}.xml")

def get_model(model_name : str, main_folder : str):
    gem_file = get_gem_file(model_name, main_folder)
    try:
        model = read_sbml_model(gem_file)
        return model
    except:
        return None

def get_model_from_sample_file(sample_file : str, main_folder : str):
    """Get the cobra model for a geven sample file name"""
    model_name = get_model_name_from_sample_file(sample_file)
    return get_model(model_name, main_folder)

def get_reaction_name(reaction):
    reaction_parts = [f'{m.name}({m.compartment})[{reaction.metabolites[m]}]' for m in reaction.metabolites]
    name = "".join(sorted(reaction_parts))
    return name.replace(",", ".")

def get_rename_dict(model) -> dict[str,str]:
    """Gets the renaming dict for a given file.
    
    This will transform the file path to get the name and path for the reactions.
    Then the sbml model loaded will be used to generate a dictionary mapping metabolite
    names to 'reaction_names'.
    
    Arguments:
        path: The path of the sample which will be used to find a model.

    Returns:
        renaming: A dictionary mapping metabolits names to 'reaction_names'
    """
    if model == None:
        return None
    return {r.id : get_reaction_name(r) for r in model.reactions}

def get_non_zero_columns(df : pd.DataFrame) -> list[str]:
    """Get the non-zero column names from a DataFrame."""
    non_zeros = np.any(df.values != 0.0, axis=0)
    return df.columns[non_zeros]

def make_tmp(path : str, n : int, source_df : pd.DataFrame, random_state : RandomState):
    """Splits of a tmp file from source df of size n to be stored in path."""
    sample = source_df.sample(min(n, len(source_df.index)), random_state=random_state)
    source_df.drop(index=sample.index, inplace=True)
    with open(path, 'wb') as file: pickle.dump(sample, file)

def get_reactions_in_compartments(models : list[any], compartments : list[str]) -> list[str]:
    reactions = set()
    for m in models:
        reactions = reactions.union({get_reaction_name(r) for r in m.reactions})
    return reactions

def get_random_state(seed, index=0):
    index %= 2**32
        
    if seed == None:
        return RandomState(random.randint(0, 2**32))
    return RandomState(seed + index)

class FluxDataset(Dataset):
    '''Class alowing a fluxdataset.csv file to be loaded into pytorch.'''
    def __init__(self, 
                 path : str, 
                 dataset_size : int = DEFAULT_DATASET_SIZE,
                 model_folder : str = None,
                 join : str ='inner', 
                 verbose : bool = True, 
                 reload_aux : bool = False, 
                 skip_tmp : bool = False,
                 columns : list[str] = None,
                 compartments : list[str] = None,
                 seed : int = None):
        '''Takes files - a path to a csv file containing the data to be leaded. 
        
        The data is automatically normalized when loaded.
        
        Args:
            path: The path (.csv file or folder containing .csv files).
            dataset_size: The size'''
        self.set_folder(path)
        self.model_folder = model_folder if model_folder != None else self.folder
        self.join = join
        self.verbose = verbose
        self.reload_aux = reload_aux
        self.compartments = compartments
        self.seed = seed
        self.dataset_size = dataset_size

        # Find renamings and joins
        self.load_models()
        self.find_renaming()
        self.find_joins(self.files)
        
        if columns == None:
            columns = self.joins[join]
        elif not any([m == None for m in self.models.values()]):
            reactions_in_compartments = get_reactions_in_compartments(self.models.values(), compartments)
            columns = list(set(columns).intersection(reactions_in_compartments))
        self.columns = columns

        # Load data into current
        if not skip_tmp:
            self.create_tmp_archive()
        self.load_sample()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if type(idx) == str:
            return self.normalized_values[np.array(self.labels) == idx]
        return self.labels[idx], torch.Tensor(self.normalized_values[idx])
    
    def set_folder(self, path : str):
        """Sets up the folder for the FluxDataset.
        
        Just readies the directory creating a pkl_folder

        Args:
            path: The path to a csv file or folder constaining many csv files.

        Raises:
            FileNotFoundError: If the folder doesn't exist, or there are no files under
            the path.
        
        """
        self.path = path
        self.folder = os.path.dirname(path)

        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"The folder {self.folder} does not exist and there is no flux data to load.")

        self.pkl_folder = os.path.join(self.folder, PKL_FOLDER)
        self.train_pkl_folder = os.path.join(self.pkl_folder, "train")
        self.join_path = os.path.join(self.folder, JOIN_FILE)

        self.files = [path] if path.endswith('.csv') else [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]
        self.files = {get_file_name_from_sample_file(f) : f for f in self.files}

        if len(self.files) == 0:
            raise FileNotFoundError(f"The path {self.path} has no .csv files in it.")

        if not os.path.exists(self.pkl_folder):
            os.makedirs(self.pkl_folder)
        
    def load_models(self):
        print("Loading models...")

        models_pkl_path = os.path.join(self.model_folder, PKL_FOLDER, MODELS_PKL_FILE)
        self.models = {}

        # Try Loading Via Pickle Cache
        if not self.reload_aux:
            try:
                with open(models_pkl_path, 'rb') as models_pkl_file:
                    self.models = pickle.load(models_pkl_file)
            except:
                print(f"Failed to load {models_pkl_path}")
                pass
        
        model_names = set([get_model_name_from_file_name(fn) for fn in self.files])
        for model_name in tqdm(model_names, desc="Loading Models", disable=not self.verbose):
            if model_name not in self.models:
                model = get_model(model_name, self.model_folder)
                self.models[model_name] = model

        print(f"Models loaded:")
        for model_name, model in self.models.items():
            print(f"    {model_name} : {'not ' if model == None else ''} found.")

        if self.model_folder == self.folder:
            with open(models_pkl_path, 'wb') as models_pkl_file:
                pickle.dump(self.models, models_pkl_file)

    def find_renaming(self):
        """Loads the renaming for all metabolites in the samples."""
        self.renaming_dicts = {}
        for model_name, model in tqdm(self.models.items(), desc='Creating Renaming', disable=not self.verbose):
            gem_rename_dict = get_rename_dict(model)
            if gem_rename_dict != None:
                self.renaming_dicts[model_name] = gem_rename_dict 

    def get_pkl_path(self, file_name : str) -> str:
        """Get the pkl path for a given name.
        
        Arguments:
            name: The name for the pkl path.

        Returns:
            path: The path to the pkl file.
        """
        return os.path.join(self.pkl_folder, f"{file_name}.pkl")

    def get_df(self, file_name : str, n : int = None) -> pd.DataFrame:
        """Loads a DataFrame for a given name.
        
        Arguments:
            name: The name of the sample

        Returns:
            df: The dataframe of a sample for the given name.
        """
        pkl_path = self.get_pkl_path(file_name)
        csv_path = self.files[file_name]
        model_name = get_model_name_from_file_name(file_name)

        if n != None:
            df = pd.read_csv(csv_path, index_col=0, nrows=n)
        else:
            try:
                with open(pkl_path, 'rb') as pkl_file:
                    df = pickle.load(pkl_file)
            except:
                df = pd.read_csv(csv_path, index_col=0, engine='pyarrow')
                with open(pkl_path, 'wb') as pkl_file:
                    pickle.dump(df, pkl_file)

        if model_name in self.renaming_dicts:
            df = df.rename(columns=self.renaming_dicts[model_name])
            df = df.groupby(df.columns, axis=1).agg(sum)
            return df
        else:
            return df
    
    def find_joins(self, files : list[str]):
        """Finds the different joints for the dataset.
        
        Args:
            files: The files which the join will be defined over.
            
        Raises:

        """
        inner, outer = set(), set()

        for i, name in tqdm(enumerate(files), desc="Making inter_union", disable=not self.verbose):
            columns = get_non_zero_columns(self.get_df(name))
            inner = set(columns) if i == 0 else inner.intersection(columns)
            outer = outer.union(columns)

        self.joins = {
            'inner' : list(inner),
            'outer' : list(outer)
        }

    def create_tmp_archive(self):
        """Makes tmp files to be used to speed up sample loading.
        
        tmp files - These are pickled pd.DataFrame random subsets of the files loaded. 
        """
        samples_per_file = self.dataset_size // len(self.files)
        ensure_exists(self.train_pkl_folder)

        for name in tqdm(self.files, desc='Clearing tmp archive'):
            self.remove_tmp_files(name)

        min_sample_size = samples_per_file
        for name in tqdm(self.files, desc='Making tmps', disable=not self.verbose):
            sample_size = self.make_tmp_files(name, samples_per_file)
            min_sample_size = min(sample_size, min_sample_size)

        if min_sample_size < samples_per_file:
            new_dataset_size = min_sample_size * len(self.files)
            print(f"Dataset size too big! min_sample_size {min_sample_size}," + 
                  f" updating dataset_size {self.dataset_size} -> {new_dataset_size}")
            self.dataset_size = new_dataset_size

    def remove_tmp_files(self, file_name):
        "Removes tmp files for a certain file name."
        [rm(joinp(self.train_pkl_folder, f)) for f in os.listdir(self.train_pkl_folder) if f.startswith(file_name)]

    def make_tmp_files(self, name : str, samples_per_file : int):
        """Makes tmp files for a specific csv file."""
        df = self.get_df(name)
        if len(df.index) < samples_per_file:
            samples_per_file = len(df.index)
        
        rs = get_random_state(self.seed)

        n_saved = 0 
        while len(df.index) > 0.8 * samples_per_file:
            make_tmp(joinp(self.train_pkl_folder, f"{name}_{n_saved}.pkl"), samples_per_file, df, rs)
            n_saved += 1
            if self.seed != None:
                break

        return samples_per_file

    def load_tmp_file(self, name : str):
        """Loads a tmp file for a given name.
        
        Args:
            name: The name of the for which the tmp will be loaded.

        Raises:
            FileNotFoundError: If there is no tmp file for name.
        """
        folder = self.train_pkl_folder
        paths = [joinp(folder, f) for f in os.listdir(folder) if f.startswith(name)]

        if len(paths) == 0:
            raise FileNotFoundError(f"No tmp files found for {name}.")

        rs = get_random_state(self.seed, hash(name))

        path = paths[rs.randint(0, len(paths))]
        with open(path, 'rb') as file:
            return pickle.load(file)

    def load_dataFrame(self, df : pd.DataFrame) -> None:
        """Loads in and normalizes a dataFrame."""      
        df_num = df.select_dtypes(include='number')
        df_norm = (df_num-df_num.mean())/df_num.std()
        df_norm.fillna(0, inplace=True)
        
        self.data = df
        self.normalized_values = df_norm.values
        self.labels = list(df['label'].values)
        self.unique_labels = list(set(self.labels))

    def load_sample(self) -> None:
        """Loads a sample into the dataset.
        """
        sections = [pd.DataFrame(columns=self.columns + ['label'])]
        for file_name in tqdm(self.files, desc='Loading sample', disable=not self.verbose):
            model_name = get_model_name_from_file_name(file_name)
            tmp_sample_df = self.load_tmp_file(file_name)
            tmp_sample_df['label'] = model_name
            sections.append(tmp_sample_df)
        
        df = pd.concat(sections, ignore_index=True, sort=False, join='outer').fillna(0)
        df = df[df.columns.intersection(self.columns + ['label'])]
        self.load_dataFrame(df)

    def set_columns(self, columns : list[str]):
        self.columns = columns

        df_data = {c : self.data[c] for c in columns if c in self.data.columns}
        df = pd.DataFrame(df_data, columns=columns + ['label'])
        df.fillna(0, inplace=True)

        self.data = df
        self.normalized_values = np.array(df.drop(columns='label').values.astype(float))
        self.normalized_values


