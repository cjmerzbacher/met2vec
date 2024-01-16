import os
import re
import pickle
import pandas as pd
import numpy as np

from misc.constants import *
from misc.io import *
from reproducability import get_random_state, RandomState
from fluxModel import FluxModel


def get_file_name(path : str):
    return 

def get_model_name(file_name : str):
    end_pattern = r'_([0-9]|k)+(\([0-9]*\))?$'
    return re.sub(end_pattern, '', file_name)

def get_gem_file(model_name : str, folder : str):
    return os.path.join(folder, 'gems', f"{model_name}.xml")

def make_tmp(path : str, n : int, source_df : pd.DataFrame, random_state : RandomState, model_name : str):
    """Splits of a tmp file from source df of size n to be stored in path."""
    sample = source_df.sample(min(n, len(source_df.index)), random_state=random_state)
    source_df.drop(index=sample.index, inplace=True)
    with open(path, 'wb') as file: pickle.dump(sample, file)


class FluxFile:
    def __init__(self, path : str, model_main_folder : str = None, model : FluxModel = None, seed : int = None):
        self.path = path    
        self.model = model
        self.seed = seed

        self.basename = os.path.basename(path)
        self.file_name = os.path.basename(path).removesuffix(".csv")
        self.main_folder = os.path.dirname(path)
        
        self.pkl_folder = join(self.main_folder, PKL_FOLDER, make_folder=True)
        self.pkl_path = join(self.pkl_folder, f"{self.file_name}.pkl")

        self.train_pkl_folder = join(self.pkl_folder, TRAIN_FOLDER, make_folder=True)

        self.model_folder = model_main_folder
        if self.model_folder is None:
            self.model_folder = self.main_folder

        self.model_name = get_model_name(self.file_name)

    def get_nonzero_columns(self):
        df = self.get_df()

        non_zeros = np.any(df.values != 0.0, axis=0)
        return df.columns[non_zeros]

    def make_df_pkl(self):
        df = pd.read_csv(self.path)
        safe_pkl_dump(self.pkl_path, df)
        return df

    def get_df(self):
        df = safe_pkl_load(self.pkl_path)
        if df is None:
            df = self.make_df_pkl()

        if self.model != None:
            renaming = self.model.get_renaming_dict()
            df.rename(columns=renaming, inplace=True)
            df = df.groupby(df.columns, axis=1).agg(sum)

        return df
    
    def set_model(self, model : FluxModel):
        self.model = model
    
    def get_rs(self, index=0):
        return get_random_state(self.seed, index)
    
    def make_tmps(self, samples_per_file : int) -> int:
        df = self.get_df()

        if len(df.index) < samples_per_file:
            samples_per_file = len(df.index)
            n = 1
        else:
            n = len(df.index) // samples_per_file + 1

        rs = self.get_rs()

        for i in range(n):
            make_tmp(
                os.path.join(
                    self.train_pkl_folder,
                    f"{self.file_name}_{i}.pkl"
                ),
                samples_per_file,
                df, rs, 
                self.model_name,
            )

            if self.seed is not None:
                break

        return samples_per_file
    
    def load_tmp_file(self):
        paths = [
            os.path.join(self.pkl_folder, f)
            for f in os.listdir(self.pkl_folder)
            if f.startswith(self.file_name)
        ]

        rs = self.get_rs()
        path = paths[rs.randint(0, len(paths))]
        with open(path, 'rb') as file:
            return pickle.load(file)
        

