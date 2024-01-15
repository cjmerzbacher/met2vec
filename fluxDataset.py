import pandas as pd
import numpy as np
import torch
import os
import logging

from torch.utils.data import Dataset
from tqdm import tqdm
from misc.constants import *
from misc.io import *

from fluxFile import FluxFile
from fluxModel import FluxModel

logging.getLogger('cobra').setLevel(logging.CRITICAL)

GEM_FOLDER = 'gems'
RENAME_DICT_FILE = ".renaming.json"
JOIN_FILE = ".join.json"
PKL_FOLDER = ".pkl"
DEFAULT_DATASET_SIZE = 65536

rm = os.unlink
joinp = os.path.join
ensure_exists = lambda f: None if os.path.exists(f) else os.makedirs(f)

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
                 seed : int = None):
        '''Takes files - a path to a csv file containing the data to be leaded. 
        
        The data is automatically normalized when loaded.
        
        Args:
            path: The path (.csv file or folder containing .csv files).
            dataset_size: The size'''
        self.set_folder(path, model_folder)
        self.load_flux_files()

        self.join = join
        self.verbose = verbose
        self.reload_aux = reload_aux
        self.seed = seed
        self.dataset_size = dataset_size

        # Find renamings and joins
        self.load_models()
        self.find_joins()
        
        if columns == None:
            columns = self.joins[join]
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
    
    def set_folder(self, path : str, model_folder : str):
        """Sets up the folder for the FluxDataset.
        
        Just readies the directory creating a pkl_folder

        Args:
            path: The path to a csv file or folder constaining many csv files.

        Raises:
            FileNotFoundError: If the folder doesn't exist, or there are no files under
            the path.
        
        """
        self.path = path
        self.main_folder = path if os.path.isdir(path) else os.path.dirname(path)
        self.model_folder = model_folder if model_folder != None else self.main_folder

        if not os.path.exists(self.main_folder):
            raise FileNotFoundError(f"The folder {self.main_folder} does not exist and there is no flux data to load.")

    def load_flux_files(self):   
        if self.path.endswith('.csv'):     
            flux_paths = [self.path] 
        else:
            flux_paths = [
                os.path.join(self.path, f) 
                for f in os.listdir(self.path) 
                if f.endswith('.csv')
            ]

        if len(flux_paths) == 0:
            raise FileNotFoundError(f"'{self.path}' has no .csv files.")

        self.flux_files : dict[str, FluxFile] = {}
        for f in flux_paths:
            flux_file = FluxFile(f, self.model_folder) 
            self.flux_files[flux_file.file_name] = flux_file

    def load_models(self):
        self.flux_models : dict[str, FluxModel] = {}

        flux_file_by_model_name = {}
        for ff in self.flux_files.values():
            model_name = ff.model_name
            if model_name not in flux_file_by_model_name:
                flux_file_by_model_name[model_name] = []
            flux_file_by_model_name[model_name].append(ff)

        for model_name, flux_files in flux_file_by_model_name.items():
            try:
                fm = FluxModel(model_name, self.model_folder)
            except:
                fm = None

            self.flux_models[model_name] = fm 
            for ff in flux_files:
                ff.set_model(fm)


    def find_joins(self):
        inner, outer = set(), set()

        for i, ff in tqdm(enumerate(self.flux_files.values()), desc="Making inter_union"):
            columns = ff.get_nonzero_columns()
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
        samples_per_file = self.dataset_size // len(self.flux_files)

        min_sample_size = samples_per_file
        for ff in tqdm(self.flux_files.values(), desc='Making tmps'):
            sample_size = ff.make_tmps(samples_per_file)
            min_sample_size = min(sample_size, min_sample_size)

        if min_sample_size < samples_per_file:
            new_dataset_size = min_sample_size * len(self.flux_files)
            print(f"Dataset size too big! min_sample_size {min_sample_size}," + 
                  f" updating dataset_size {self.dataset_size} -> {new_dataset_size}")
            self.dataset_size = new_dataset_size

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
        columns = self.columns + ['label']
        sections = [pd.DataFrame(columns=columns)]
        for ff in tqdm(self.flux_files.values(), desc='Loading sample'):
            sample = ff.load_tmp_file()
            sample['label'] = ff.model_name
            sections.append(sample)
        
        df = pd.concat(sections, ignore_index=True, sort=False, join='outer').fillna(0)
        df = df[df.columns.intersection(columns)]
        self.load_dataFrame(df)

    def set_columns(self, columns : list[str]):
        self.columns = columns

        df_data = {c : self.data[c] for c in columns if c in self.data.columns}
        df = pd.DataFrame(df_data, columns=columns + ['label'])
        df.fillna(0, inplace=True)

        self.data = df
        self.normalized_values = np.array(df.drop(columns='label').values.astype(float))
        self.normalized_values


