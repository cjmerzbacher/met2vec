import pandas as pd
import re
import os

#def prepare_section(df : pd.DataFrame) -> pd.DataFrame:
#    df.drop(columns=df.columns[0:1], inplace=True)
#
#    all_zeros = (df == 0).all()
#    zero_columns = [n for n in df.columns if all_zeros[n]]
#    mean = df.mean()
#    std = df.std()
#    std[zero_columns] = 1.0
#
#    return (df - mean) / std, mean, std

def prepare_section(df : pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=df.columns[0:1], inplace=True)
    return df

def get_label_from_path(path : str) -> str:
    return re.match(r"([a-z_]*)[a-z]", os.path.basename(path)).group()


class PlottingDataset:
    def __init__(self):
        self.sections = {}
        self.df = pd.DataFrame()

    def __len__(self):
        return len(self.df)

    def add_section(self, path):
        label = get_label_from_path(path)

        df = prepare_section(pd.read_csv(path))
        df['label'] = label

        self.sections[label] = df

        self.df = pd.concat([self.df, df])
        self.df.fillna(0, inplace=True)

    def values(self):
        return self.df.drop(columns=['label']).values
    
    def normalized_values(self):
        df = self.df.drop(columns=['label'])

        all_zeros = (df == 0).all()
        zero_columns = [n for n in df.columns if all_zeros[n]]
        mean = df.mean()
        std = df.std()
        std[zero_columns] = 1.0

        return ((df - mean) / std).values
    
    def columns(self):
        return self.df.columns

    def shape(self):
        return self.df.shape

