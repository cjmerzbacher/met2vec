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
from fluxModel import *

logging.getLogger('cobra').setLevel(logging.CRITICAL)

GEM_FOLDER = 'gems'
RENAME_DICT_FILE = ".renaming.json"
JOIN_FILE = ".join.json"
PKL_FOLDER = ".pkl"
DEFAULT_DATASET_SIZE = 65536

rm = os.unlink
joinp = os.path.join
ensure_exists = lambda f: None if os.path.exists(f) else os.makedirs(f)

def get_conversion_matrix(from_reactions : list[str], to_reactions : list[str]):
    def row(from_reaction):
        row = np.zeros(len(to_reactions))
        if from_reaction in to_reactions:
            from_i = to_reactions.index(from_reaction)
            row[from_i] = 1
        return row

    return np.array([
        row(to_reaction) 
        for to_reaction in from_reactions
    ])
    
class FluxDataset(Dataset):
    '''Class alowing a fluxdataset.csv file to be loaded into pytorch.'''
    def __init__(self, 
                 path : str, 
                 n : int = DEFAULT_DATASET_SIZE,
                 model_folder : str = None,
                 seed : int = None):
        '''Takes files - a path to a csv file containing the data to be leaded. 
        
        The data is automatically normalized when loaded.
        
        Args:
            path: The path (.csv file or folder containing .csv files).
            dataset_size: The size'''
        self.seed = seed
        self.n = n

        self.set_folder(path, model_folder)

        print(f"Creating dataset from {self.main_folder} with size {self.n} and seed={seed}")

        self.find_flux_files()
        self.find_models()
        self.load_fluxes()

        # Load data into current
        self.reload_sample()
        self.create_stoicheometric_matrix()

        self.C = self.get_conversion_matrix(self.core_reaction_names)

        print(f"inner size -> {len(self.core_reaction_names)}")
        print(f"outer size -> {len(self.reaction_names)}")

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
        self.model_folder = model_folder 
        
        if model_folder is None:
            print(f"No model folder specified using {self.main_folder}")
            self.model_folder = self.main_folder

        if not os.path.exists(self.main_folder):
            raise FileNotFoundError(f"The folder {self.main_folder} does not exist and there is no flux data to load.")

    def find_flux_files(self):   
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
            flux_file = FluxFile(f, self.model_folder, seed=self.seed) 
            self.flux_files[flux_file.file_name] = flux_file

    def find_models(self):
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

            if fm is None:
                continue

            self.flux_models[model_name] = fm 
            for ff in flux_files:
                ff.set_model(fm)

    def load_fluxes(self):
        samples_per_file = self.n // len(self.flux_files)
        min_spf = samples_per_file

        flux_sums = []
        flux_suqare_sums = []
        flux_columns = []
        n_fluxes = 0

        for ff in tqdm(self.flux_files.values(), desc="Loading fluxes..."):
            flux_df = ff.get_df()
            num_df = flux_df.drop(columns=flux_df.columns.intersection(SOURCE_COLUMNS))

            act_spf = ff.make_tmps(samples_per_file, flux_df)
            min_spf = min(act_spf, min_spf)

            #flux_columns.append(flux_df.columns[np.any(flux_df.values != 0, axis=0)])
            flux_columns.append(set(num_df.columns))

            flux_sums.append(num_df.sum())
            flux_suqare_sums.append((num_df**2).sum())
            n_fluxes += len(num_df.index)

        print(f"{n_fluxes} fluxes loaded...")

        self.reaction_names = sorted(set.union(*flux_columns))
        self.core_reaction_names = sorted(set.intersection(*flux_columns))

        self.flux_mean = (pd.DataFrame(flux_sums) / n_fluxes).fillna(0).sum().reindex(self.reaction_names)

        mean_squared = (self.flux_mean ** 2)
        square_mean = pd.DataFrame(flux_suqare_sums).fillna(0).sum()
        self.flux_std = np.sqrt((square_mean / n_fluxes) - mean_squared).fillna(0).reindex(self.reaction_names)

        if min_spf < samples_per_file:
            new_n = min_spf * len(self.flux_files)
            print(f"n too big! min_spf {min_spf}, updating n: {self.n} -> {new_n}")
            self.n = new_n

    def get_mu_std(self):
        return self.flux_mean.values, self.flux_std.values
    
    def normalize_data(self, data : pd.DataFrame) -> pd.DataFrame:
        df_num = data.drop(columns=SOURCE_COLUMNS).select_dtypes(include='number')
        df_norm = ((df_num - self.flux_mean) / self.flux_std).fillna(0)
        return df_norm
    
    def get_normalized_data(self, fluxes : list[str] = None):
        nd = self.normalize_data(self.data)
        nd[SOURCE_COLUMNS] = self.data[SOURCE_COLUMNS]

        if fluxes is None:
            fluxes = self.reaction_names 

        nd = nd.drop(columns = nd.columns.difference(SOURCE_COLUMNS + fluxes))
        nd = nd.reindex(columns = SOURCE_COLUMNS + fluxes) 

        return nd

    def load_dataFrame(self, df : pd.DataFrame) -> None:
        """Loads in and normalizes a dataFrame."""      
        self.data = df
        self.normalized_values = self.normalize_data(df).values
        self.labels = list(df[LABEL].values)
        self.unique_labels = df[LABEL].unique()

    def reload_sample(self) -> None:
        """Loads a sample into the dataset.
        """
        columns = list(set(self.reaction_names + SOURCE_COLUMNS))
        samples = [pd.DataFrame(columns=columns)]
        flux_files_it = list(enumerate(self.flux_files.values()))

        for i, ff in tqdm(flux_files_it, desc='Loading sample'):
            sample = ff.load_tmp_file()
            sample[FILE_N] = i

            samples.append(sample)
        
        df = pd.concat(samples, ignore_index=True).fillna(0)
        df = df[df.columns.intersection(columns)]
        df = df.reindex(columns=self.reaction_names + SOURCE_COLUMNS)
        self.load_dataFrame(df)

    def create_stoicheometric_matrix(self):
        if len(self.flux_models) == 0:
            return

        models = [
            flux_model.get_cobra_model()
            for flux_model in tqdm(list(self.flux_models.values()), desc="Loading models")
        ]

        self.metabolite_names = list(set.union(*[
            set(map(get_metabolite_name, model.metabolites))
            for model in models
        ]))

        reactions = {
            get_reaction_name(reaction) : reaction
            for reaction in sum([
                model.reactions
                for model in models
            ], start=[])
        }

        S_raw = np.zeros((
            len(self.metabolite_names),
            len(self.reaction_names)
        ))

        r_ind = self.reaction_names.index
        m_ind = self.metabolite_names.index

        for reaction_name, reaction in reactions.items():
            for metabolite, stoich in reaction.metabolites.items():
                metabolite_name = get_metabolite_name(metabolite)

                S_raw[
                    m_ind(metabolite_name),
                    r_ind(reaction_name)
                ] = stoich

        self.S = pd.DataFrame(
            S_raw,
            columns = self.reaction_names,
        )
        self.S[METABOLITE] = self.metabolite_names
        self.S = self.S.set_index(METABOLITE)

    def get_conversion_matrix(self, to_reactions):
        return get_conversion_matrix(self.reaction_names, to_reactions)

