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

def get_n_temps(n_samples, samples_per_file):
    if n_samples <= samples_per_file:
        samples_per_file = n_samples
        return 1
    else:
        return n_samples // samples_per_file + 1


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

    def get_columns(self, non_zero=False):
        df = self.get_df()
        df.drop(columns=df.columns.intersection(SOURCE_COLUMNS), inplace=True)

        if non_zero:
            non_zeros = np.any(df.values != 0.0, axis=0)
            return df.columns[non_zeros]
        else:
            return df.columns
        

    def make_df_pkl(self):
        try:
            df = pd.read_csv(self.path,index_col=0)
        
            safe_pkl_dump(self.pkl_path, df)
            return df
        except Exception as e:
            raise Exception(f"Failed to read df for {self.path}.").with_traceback(e.__traceback__)

    def get_df(self):
        df = safe_pkl_load(self.pkl_path)
        if df is None:
            df = self.make_df_pkl()

        df.drop(columns=df.columns.intersection(["Unnamed: 0"]), inplace=True)

        # Source Details
        df.reset_index(inplace=True)
        df.rename(columns = {'index':SAMPLE_N}, inplace=True)
        df[FILE_NAME] = self.file_name
        df[LABEL] = self.model_name

        if self.model != None:
            renaming = self.model.get_renaming_dict()
            df.rename(columns=renaming, inplace=True)
            df = df.groupby(df.columns, axis=1).agg(sum)

        return df
    
    def set_model(self, model : FluxModel):
        self.model = model
    
    def get_rs(self, index=0):
        return get_random_state(self.seed, index)
    
    def make_tmps(self, samples_per_file : int, df : pd.DataFrame=None) -> int:
        if df is None:
            df = self.get_df()

        n = get_n_temps(len(df.index), samples_per_file)
        rs = self.get_rs()
        self.clear_tmp_files()

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
    
    def clear_tmp_files(self):
        paths = self.get_tmp_paths()
        for path in paths:
            os.unlink(path)

    def get_tmp_paths(self):
        return [
            os.path.join(self.train_pkl_folder, f)
            for f in os.listdir(self.train_pkl_folder)
            if f.startswith(self.file_name)
        ]
    
    def load_tmp_file(self):
        paths = self.get_tmp_paths()
        rs = self.get_rs()
        path = paths[rs.randint(0, len(paths))]
        with open(path, 'rb') as file:
            return pickle.load(file)
        

