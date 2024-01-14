import os
import re
import pickle
import pandas as pd

from misc.constants import *
from reproducability import get_random_state

TRAIN = 'train'

def get_file_name(path):
    return os.path.basename(path).removesuffix(".csv")

def get_model_name(file_name):
    end_pattern = r'_([0-9]|k)+(\([0-9]*\))?$'
    return re.sub(end_pattern, '', file_name)

def get_gem_file(model_name, folder):
    return os.path.join(folder, 'gems', f"{model_name}.xml")

def make_tmp(path : str, n : int, source_df : pd.DataFrame, random_state : RandomState):
    """Splits of a tmp file from source df of size n to be stored in path."""
    sample = source_df.sample(min(n, len(source_df.index)), random_state=random_state)
    source_df.drop(index=sample.index, inplace=True)
    with open(path, 'wb') as file: pickle.dump(sample, file)


class FluxFile:
    def __init__(self, path, model_main_folder=None, models=None, seed=None):
        self.path = path    
        self.models = models
        self.seed = seed

        self.basename = os.path.basename(path)

        self.main_folder = os.path.dirname(path)
        self.pkl_folder = os.path.join(self.main_folder, PKL_FOLDER)
        self.train_pkl_folder = os.path.join(self.pkl_folder, TRAIN)

        self.model_folder = model_main_folder
        if self.model_folder is None:
            self.model_folder = self.main_folder

        self.file_name = get_file_name(path)
        self.model_name = get_model_name(self.file_name)
        self.gem_file = get_gem_file(self.model_name, self.model_folder)

    def get_df(self):
        #pkl_path = os.path.join(self.pkl_folder, self.basename)
        #
        #try:
        #    with open(pkl_path, 'rb') as pkl_file:
        #        df = pickle.load(pkl_file)
        #except:
        #    df = pd.read_csv(self.path)
        #    try:
        #        with open(pkl_path, 'wb') as pkl_file:
        #            pickle.dump(df, pkl_file)
        #    except:
        #        print(f"Unable to write pkl_file '{pkl_path}' for {self.file_name}.")

        df = pd.read_csv(self.path)

        if self.models != None:
            assert NotImplementedError()

        return df
    
    def get_rs(self, index=0):
        return get_random_state(self.seed, index)
    
    def make_tmps(self, samples_per_file):
        df = self.get_df()

        if len(df.index) < samples_per_file:
            samples_per_file = len(df.index)
            n = 1
        else:
            n = len(df.index) // samples_per_file + 1

        rs = self.get_rs()

        for i in range(n):
            make_tmp(os.path.join(
                self.train_pkl_folder,
                f"{self.file_name}_{i}.pkl",
                samples_per_file,
                df, rs
            ))

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
