import argparse
import pandas as pd
import logging
import os

from pprint import pp
from cobra.io import read_sbml_model

logging.getLogger('cobra').setLevel(logging.ERROR)

def read_arguments():
    parser = argparse.ArgumentParser("Flux Renamer", "Basic tool used to rename the columns in a flux sample file to compound names instead of metaids.")
    parser.add_argument("-e", "--edit", required=True, type=str)
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-o", "--output", default=None, type=str)

    args = parser.parse_args()
    if args.output is None:
        args.output = args.edit

    return args

def get_reaction_name(reaction):
    name = ''
    metabolites = reaction.metabolites
    for metabolite in sorted([m for m in metabolites], key=lambda m: m.name):
        name += f'{metabolite.name}[{metabolites[metabolite]}]'

    return name.replace(",", ".")

def get_raname_dict(model):
    return {r.id : get_reaction_name(r) for r in model.reactions}

def rename_flux_df(flux_df, model):
    rename_dict = get_reaction_name(model)
    return flux_df.rename(columns=rename_dict)


def main():
    args = read_arguments()
    model = read_sbml_model(args.model)
    flux_df = pd.read_csv(args.edit)
    flux_df = rename_flux_df(flux_df, model)

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    flux_df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
