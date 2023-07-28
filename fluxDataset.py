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
        self.ignore = np.all(self.data == 0.0, axis=0, keepdims=True)

        self.mean = np.mean(self.data, axis=0)
        self.data = (self.data - self.mean)

        self.std = np.std(self.data, axis=0)
        self.data = self.data / self.std

        self.data[np.repeat(self.ignore, self.data.shape[0], axis=0)] = 0.0


fd = FluxDataset("./data/samples/liver_100.csv")
