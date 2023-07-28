import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FluxDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.data = pd.read_csv(path).values
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])
