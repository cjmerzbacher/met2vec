import pandas as pd
import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset

class FluxDataset(Dataset):
    '''Class alowing a fluxdataset.csv file to be loaded into pytorch.'''
    def __init__(self, path):
        '''Takes files - a path to a csv file containing the data to be leaded. The data is automatically normalized when loaded.'''
        self.files = [path] if '.csv' in path else [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]
        
        self.columns = set()
        for file in self.files:
            df = pd.read_csv(file, nrows=0)
            df.drop(columns=df.columns[0], inplace=True)
            self.columns = self.columns.union(df.columns)
        self.columns = list(self.columns)

        self.reload_mix()

    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])

    def reload_mix(self):
        self.file = random.sample(self.files, 1)[0]
        
        df = pd.read_csv(self.file)
        df.drop(df.columns[0], axis=1, inplace=True) # First column is counter
        df = pd.concat([pd.DataFrame(columns=self.columns), df])
        df.fillna(0.0, inplace=True)
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
