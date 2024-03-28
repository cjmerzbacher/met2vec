import os
import re
import pickle
import math
import pandas as pd
import numpy as np
import shutil

from misc.constants import *
from misc.io import *
from reproducability import get_random_state, RandomState
from fluxModel import FluxModel
from random import randint


def get_model_name(file_name : str):
    end_pattern = r'_([0-9]|k)+(\([0-9]*\))?$'
    return re.sub(end_pattern, '', file_name)

def get_gem_file(model_name : str, folder : str):
    return os.path.join(folder, 'gems', f"{model_name}.xml")

def make_tmp(path : str, n : int, source_df : pd.DataFrame, random_state : RandomState, model_name : str):
    """Splits of a tmp file from source df of size n to be stored in path."""
    sample = source_df.sample(min(n, len(source_df.index)), random_state=random_state)
    source_df.drop(index=sample.index, inplace=True)
    if len(sample) >= 0.8 * n:
        with open(path, 'wb') as file: pickle.dump(sample, file)

def get_n_temps(n_samples, samples_per_file):
    if n_samples <= samples_per_file:
        samples_per_file = n_samples
        return 1
    else:
        return math.ceil(n_samples / samples_per_file)


class FluxFile:
    def __init__(self, path : str, model : FluxModel = None, seed : int = None):
        """FluxFile is an interface which manages loading the the large flux sample files created by flux sampling.
        
        Args:
            path: The path of the file.
            model: The model (e.g. GSM wrapper).
            seed: Seed fixes randomness set if repeatability is required.
        """
        self.path = path    
        self.model = model
        self.seed = seed

        self.basename = os.path.basename(path)
        self.file_name = os.path.basename(path).removesuffix(".csv")
        self.main_folder = os.path.dirname(path)
        
        self.pkl_folder = join(self.main_folder, PKL_FOLDER, make_folder=True)
        self.pkl_path = join(self.pkl_folder, f"{self.file_name}.pkl")

        self.pkl_folder = join(self.pkl_folder, f"{self.file_name}{randint(0, 65536)}", make_folder=True)

        self.model_name = get_model_name(self.file_name)
        self.tmp_paths_queue : list[str] = []

    def __del__(self):
        """Makes sure pkl folder is removed, if program crashes might have to be done manually."""
        shutil.rmtree(self.pkl_folder)

    def get_columns(self, non_zero=False):
        """Finds the columns of this fluxfile, these will be renamed already."""
        df = self.get_df()
        df.drop(columns=df.columns.intersection(SOURCE_COLUMNS), inplace=True)

        if non_zero:
            non_zeros = np.any(df.values != 0.0, axis=0)
            return df.columns[non_zeros]
        else:
            return df.columns
        

    def make_df_pkl(self):
        """Loads the flux .csv file and saves it as a pkl to make future loadings fater."""
        try:
            df = pd.read_csv(self.path,index_col=0)
        
            safe_pkl_dump(self.pkl_path, df)
            return df
        except Exception as e:
            raise Exception(f"Failed to read df for {self.path}.").with_traceback(e.__traceback__)

    def get_df(self):
        """Get a cope of the dataframe with renaming applied."""
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
        """Sets the model (GSM) used for renaming."""
        self.model = model
    
    def get_rs(self, index=0):
        """Gets a random state set by seed, index allow different random states to be generated."""
        return get_random_state(self.seed, index)
    
    def make_tmps(self, samples_per_file : int, df : pd.DataFrame=None) -> int:
        """Generates and saves tmp files to allow smaller samples to be loaded by FluxDataset"""
        if df is None:
            df = self.get_df()

        n = get_n_temps(len(df), samples_per_file)
                        
        samples_per_file = math.ceil(len(df) / n)
        rs = self.get_rs()

        for i in range(n):
            make_tmp(
                os.path.join(
                    self.pkl_folder,
                    f"{self.file_name}_{i}.pkl"
                ),
                samples_per_file,
                df, rs, 
                self.model_name,
            )

            if self.seed is not None:
                break

        return samples_per_file
    
    def get_tmp_paths(self):
        """Find all tmp files (saved as .pkl files)."""
        return [
            os.path.join(self.pkl_folder, f)
            for f in os.listdir(self.pkl_folder)
            if f.startswith(self.file_name)
        ]
    
    def reset_tmp_paths_queue(self):
        """Resets the tmp queue"""
        rs = self.get_rs()
        self.tmp_paths_queue = self.get_tmp_paths() 
        rs.shuffle(self.tmp_paths_queue)
    
    def load_tmp_file(self):
        """Loads the next tmp file in the tmp queue"""
        if len(self.tmp_paths_queue) == 0:
            self.reset_tmp_paths_queue()

        path = self.tmp_paths_queue.pop(0)
        with open(path, 'rb') as file:
            return pickle.load(file)
        

