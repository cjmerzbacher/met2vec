import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FluxDataset(Dataset):
    def __init__(self, path):
        self.path = path

        df = pd.read_csv(path)
        df.drop(df.columns[0], axis=1) # First column is counter

        self.data = df.values

        self.normalize()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])
    
    def normalize(self):
        ignore = np.repeat(np.all(self.data == 0.0, axis=0, keepdims=True), self.data.shape[0], axis=0)

        self.mean = np.mean(self.data, axis=0)
        self.data = (self.data - self.mean)

        self.std = np.std(self.data, axis=0)
        self.std[np.all(ignore, axis=0)] = 1.0

        self.data = self.data / self.std

        self.data[ignore] = 0.0
