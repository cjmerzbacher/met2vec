from cobra.io import read_sbml_model
from tqdm import tqdm
from pprint import pp

import argparse
import os
import re
import logging
import pandas as pd
import numpy as np

logging.getLogger('cobra').setLevel(logging.CRITICAL)

UNION = 'U'
INTER = 'I'

def get_args():
    parser = argparse.ArgumentParser('Flux Sample Combiner', 'Used to combine multiple flux samples together to a larger sample.')
    parser.add_argument('action', choices=[UNION, INTER], help=f'Control whether the resulting combined sample keeps all columns ({UNION}) or only uses the columns which all samples have non-zero values ({INTER}).')
    parser.add_argument('input', help='The input directory from which all csv files will be parsed.')
    parser.add_argument('output', help='The output directory where amended copies of all csv files will be saved.')
    parser.add_argument('--model', '-m', help='The location where genome scale models are stored in xml format.')

    return parser.parse_args()

def find_files_with_extension(args, extension):
    return [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(extension)]

def find_model_file(sample_file, args):
    try: 
        name = re.sub('_[0-9|k]*.csv', '', re.search(r'[a-zA-Z \-_,]*_[0-9|k]+.csv', sample_file).group())
        return os.path.join(args.model, f'{name}.xml')
    except AttributeError:
        print('Cannot find name from file {sample_file}')

def get_reaction_name(reaction):
    name = ''
    metabolites = reaction.metabolites
    for metabolite in sorted([m for m in metabolites], key=lambda m: m.name):
        name += f'{metabolite.name}({metabolite.compartment})[{metabolites[metabolite]}]'

    return name.replace(",", ".")

def get_rename_dict(model):
    rd = {r.id : get_reaction_name(r) for r in model.reactions}
    return rd

def read_sample(file):
    df = pd.read_csv(file)
    df.drop(columns=df.columns[0], inplace=True)
    return df

def read_sample_columns(file):
    df = pd.read_csv(file, nrows=0)
    df.drop(columns=df.columns[0], inplace=True)
    return df.columns

def c(prompt = 'Continue (y/n)?'):
    user_input = ''
    while user_input != 'y':
        user_input = input(prompt).lower()
        if user_input == 'n': quit()

def check_model_folders(sample_files, args):
    if args.model is None:
        return
    model_files = [find_model_file(f, args) for f in sample_files]
    print('Sample -> Model mapping')
    for sf, mf in zip(sample_files, model_files):
        start = '[ ]' if os.path.exists(mf) else '[x]'    
        print(start, f'{sf} -> {mf}')
    c()
    return [get_rename_dict(read_sbml_model(f)) for f in tqdm(model_files, 'Loading rename dicts')]

def main():
    args = get_args()
    sample_files = find_files_with_extension(args, '.csv')
    rename_dicts = check_model_folders(sample_files, args)

    column_sets = []
    for i in tqdm(range(len(sample_files)), 'Finding columns to keep'):
        columns = read_sample_columns(sample_files[i])
        if rename_dicts is not None:
            columns = [rename_dicts[i][c] for c in columns]
            if len(columns) != len(set(columns)):
                print("Err: nColumns has been reduced by model rename...")
                quit()
        column_sets.append(set(columns))

    column_set = set().union(*column_sets) if args.action == UNION else set(column_sets[0]).intersection(*column_sets)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for i in tqdm(range(len(sample_files)), 'Updating files'):
        sample = read_sample(sample_files[i])
        if rename_dicts is not None:
            sample.rename(columns=rename_dicts[i], inplace=True)
        drop_columns = {c for c in sample.columns if c not in column_set}
        zero_columns = set(sample.columns[np.all(sample.values == 0, axis=0)])
        sample.drop(columns=drop_columns.union(zero_columns), inplace=True)
        sample.to_csv(os.path.join(args.output, os.path.basename(sample_files[i])))

if __name__ == '__main__':
    main()

