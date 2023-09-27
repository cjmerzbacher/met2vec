from cobra.io import read_sbml_model
from tqdm import tqdm
from pprint import pp

import argparse
import os
import re
import logging
import pandas as pd
import numpy as np

logging.getLogger('cobra').setLevel(logging.ERROR)

UNION = 'U'
INTER = 'I'

def get_args():
    parser = argparse.ArgumentParser('Flux Sample Combiner', 'Used to combine multiple flux samples together to a larger sample.')
    parser.add_argument('action', choices=[UNION, INTER], help=f'Control whether the resulting combined sample keeps all columns ({UNION}) or only uses the columns which all samples have non-zero values ({INTER}).')
    parser.add_argument('input', help='The input directory from which all csv files will be parsed.')
    parser.add_argument('output', help='The output file.')
    parser.add_argument('--model', '-m', help='The location where genome scale models are stored in xml format.')

    return parser.parse_args()

def find_files_with_extension(args, extension):
    return [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(extension)]

def find_model_file(sample_file, args):
    try: 
        name = re.sub('_[0-9]*.csv', '', re.search(r'[a-z|_]*_[0-9]+.csv', sample_file).group())
        return os.path.join(args.model, f'{name}.xml')
    except AttributeError:
        print('Cannot find name from file {sample_file}')

def get_reaction_name(reaction):
    name = ''
    metabolites = reaction.metabolites
    for metabolite in sorted([m for m in metabolites], key=lambda m: m.name):
        name += f'{metabolite.name}[{metabolite.compartment}][{metabolites[metabolite]}]'

    return name.replace(",", ".")

def get_rename_dict(model):
    rd = {r.id : get_reaction_name(r) for r in model.reactions}
    return rd

def update_sample_column_names(sample, model, inplace=False):
    columns_rename_dict = get_rename_dict(model)
    return sample.rename(columns=columns_rename_dict, inplace=inplace)

def get_zero_columns(df : pd.DataFrame) -> list[str]:
    all_zeros = np.all(df.values == 0, axis=0)
    return df.columns[all_zeros]

def read_sample(file):
    df = pd.read_csv(file)
    df.drop(columns=df.columns[0], inplace=True)
    return df

def c(prompt = 'Continue (y/n)?'):
    user_input = ''
    while user_input != 'y':
        user_input = input(prompt).lower()
        if user_input == 'n': quit()

def rename_samples_columns(args, sample_files, samples):
    model_files = [find_model_file(f, args) for f in sample_files]

    print('Sample -> Model mapping')
    for sf, mf, in zip(sample_files, model_files):
        print(f'    {sf} -> {mf}')
    c()

    models = [read_sbml_model(f) for f in tqdm(model_files, 'Loading models')]
    for sample, model in zip(samples, models):
        update_sample_column_names(sample, model, inplace=True)

def main():
    args = get_args()
    sample_files = find_files_with_extension(args, '.csv')
    
    samples = [read_sample(f) for f in tqdm(sample_files, 'Loading samples')]

    if args.model is not None:
        rename_samples_columns(args, sample_files, samples)
        
    for sample in samples:
        zero_columns = set(get_zero_columns(sample))
        sample.drop(columns=zero_columns, inplace=True)

    combined_df = pd.concat(samples, join = 'inner' if args.action == INTER else 'outer')
    combined_df.fillna(0, inplace=True)

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    combined_df.to_csv(args.output)

     



if __name__ == '__main__':
    main()

