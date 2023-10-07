import argparse
import os
import json
import math
import pandas as pd
import numpy as np

from itertools import product

ARG_SETS = 'arg_sets'
HPARAMS = 'hyperparameters'
SCRIPTS_RUN = 'scripts_run'
RUNS = 'runs'
ARG_STR = 'arg_str'
ARGS = 'args'

SETUP = 'setup'
ADD = 'add'
RUN = 'run'
CLEAR = 'clear'
OUTPUT = 'output'
REMOVE = 'remove'
STATUS = 'status'

def parse_hyper_parameter(hyperparameter : list[str]):
    parser = argparse.ArgumentParser('hyperparameter')
    parser.add_argument('name', type=str)
    parser.add_argument('min', type=float)
    parser.add_argument('max', type=float)
    return parser.parse_args(hyperparameter)

def get_args():
    parser = argparse.ArgumentParser('Tuner', 'Tuner is used to test hyperparameters.')
    parser.add_argument("main_folder", type=str, help='The folder which will be used to save data during the training and tuning process.')
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # setup command
    parser_setup = subparsers.add_parser(SETUP, help='Setup folder for tuner instance to run in.')
    parser_setup.add_argument('-H' ,'--hyperparameter', action='append', nargs=3, dest='hyperparameters')

    # add command
    parser_add = subparsers.add_parser(ADD, help='Add hyperparameter to be enumerated over.')
    parser_add.add_argument('name', type=str, help='The argument/name for the hyperparameter.')
    parser_add.add_argument('--type', choices=['str', 'float', 'int'], required=True)
    parser_add.add_argument('--samples', type=int, default=10)
    parser_add.add_argument('--choices', nargs='+', type=str, default='str')
    parser_add.add_argument('--range', nargs=2, type=float)
    parser_add.add_argument('--erange', nargs=2, type=float)

    # run command
    parser_run = subparsers.add_parser(RUN, help='Run many intances of the given script to find the effect of the different hyperparameters.')
    parser_run.add_argument('script', type=str)

    # output command
    parser_output = subparsers.add_parser(OUTPUT, help='Check files made by scripts run to get results.')
    parser_output.add_argument('filename') 
    parser_output.add_argument('-a', '--average', type=int, default=10)

    # clear comand
    subparsers.add_parser(CLEAR, help='Clears the current run.')

    # remove command
    parser_remove = subparsers.add_parser(REMOVE, help='Removes hyperparameter')
    parser_remove.add_argument('name', help='The name of the hyperparameter to be removed.')

    # status command
    parser_status = subparsers.add_parser(STATUS, help='Get the status of the tuner folder.')

    return parser.parse_args() 

def setup_main_folder(main_folder):
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    

def setup_state_file(args):
    state = {HPARAMS : {}, SCRIPTS_RUN : 0, ARG_SETS: []}
    save_state_file(state, args)

def save_state_file(state, args):
    state_file_path = os.path.join(args.main_folder, 'state.json')
    with open(state_file_path, 'w') as state_file:
        json.dump(state, state_file, indent=2)

def load_state_file(args):
    state_file_path = os.path.join(args.main_folder, 'state.json')
    with open(state_file_path, 'r') as state_file:
        return json.load(state_file)

def get_range_values(_min, _max, n, _type):
    _range = _max - _min
    return [_type(_min + ((i * _range) / (n - 1))) for i in range(n)]

def get_erange_values(_min, _max, n, _type):
    _lambda = math.pow(_max / _min, 1.0 / (n - 1))
    return [_type(_min * math.pow(_lambda, i)) for i in range(n)]

def str_to_type(_str):
    match _str:
        case 'int':
            return int
        case 'float':
            return float
        case 'str':
            return str


def get_hparam_values(hp : dict[str,any]):
    _type = str_to_type(hp['type'])

    if 'range' in hp:
        return get_range_values(*hp['range'], hp['samples'], _type)
    if 'erange' in hp:
        return get_erange_values(*hp['erange'], hp['samples'], _type)
    if 'choices' in hp:
        return hp['choices']
    
def get_str_values(hp : dict[str,any]):
    values = get_hparam_values(hp)
    return [f'{v:.4e}' for v in values] if hp['type'] == 'float' else [f'{v}' for v in values]
    
def generate_arg_sets(args):
    state = load_state_file(args)
    hyperparameters : dict[str, dict] = state[HPARAMS]

    arg_possabilities = {}
    for name, h in hyperparameters.items():
        str_values = get_str_values(h)

        print(f"    {name} -> {str_values}")
        arg_possabilities[name] = str_values

    state[ARG_SETS] = []
    for i, script_values in enumerate(product(*arg_possabilities.values())):
        save_folder = os.path.join(args.main_folder, RUNS, str(i))

        script_args = []
        arg_dict = {}
        for name, value in zip(arg_possabilities.keys(), script_values):
            flag = ('-' if len(name) == 1 else '--') + name
            script_args.append(flag)
            script_args.append(value)
            arg_dict[name] = value

        arg_str = ' '.join([*script_args, save_folder])
        arg_set = {ARG_STR : arg_str, ARGS : arg_dict}

        state[ARG_SETS].append(arg_set)
    state[SCRIPTS_RUN] = 0

    save_state_file(state, args)



# commands

def setup(args):
    setup_main_folder(args.main_folder)
    setup_state_file(args)

def add(args):
    state = load_state_file(args)
    hyperparameter = {}

    def safe_add(h, a, c):
        if hasattr(a, c):
            if getattr(a, c) != None:
                h[c] = getattr(a, c)

    safe_add(hyperparameter, args, 'type')
    safe_add(hyperparameter, args, 'samples')
    safe_add(hyperparameter, args, 'choices')
    safe_add(hyperparameter, args, 'range')
    safe_add(hyperparameter, args, 'erange')

    state[HPARAMS][args.name] = hyperparameter
    save_state_file(state, args)


def run(args):
    # Generate arg sets
    state = load_state_file(args)
    if len(state[ARG_SETS]) == 0:
        generate_arg_sets(args)

    # Run Scripts
    state = load_state_file(args)
    while state[SCRIPTS_RUN] < len(state[ARG_SETS]):
        arg_set = state[ARG_SETS][state[SCRIPTS_RUN]]
        os.system(f"{args.script} {arg_set[ARG_STR]}")
        state[SCRIPTS_RUN] += 1
        save_state_file(state, args)

def clear(args):
    state : dict[str,any] = load_state_file(args)
    if ARG_SETS in state:
        state.pop(ARG_SETS)
    state[SCRIPTS_RUN] = 0
    state[ARG_SETS] = []
    save_state_file(state, args)

def output(args):
    state = load_state_file(args)

    losses_dfs = []
    for i in range(state[SCRIPTS_RUN]):
        file_path = os.path.join(args.main_folder, RUNS, str(i), 'losses.csv')
        losses_df = pd.read_csv(file_path)

        arg_set = state[ARG_SETS][i]
        for name, value in arg_set[ARGS].items():
            losses_df[f'{name}'] = value

        renamer = {c : f"run{i}_{c}" for c in losses_df.columns}
        renamer['run'] = 'run'
        losses_df.rename(columns=renamer, inplace=True)



        losses_dfs.append(losses_df)

    output_df = pd.concat(losses_dfs, axis=1)
    output_df.to_csv(args.filename, index=False)

def remove(args):
    state = load_state_file(args)

    if args.name in state[HPARAMS]:
        state[HPARAMS].pop(args.name)
    else:
        print(f"No {args.name} hyperparameter was found.")

    save_state_file(state, args)

def status(args):
    state = load_state_file(args)

    print('# Hyperparameters')
    for name, hp in state[HPARAMS].items():
        print(f"{name} -> {get_str_values(hp)}")
    if len(state[HPARAMS]) == 0:
        print("None")

    if SCRIPTS_RUN in state:
        print(f"{state[SCRIPTS_RUN]}/{len(state[ARG_SETS])} scripts run.")

    
args = get_args()

match args.command:
    case 'setup':
        setup(args)
    case 'add':
        add(args)
    case 'run':
        run(args)
    case 'clear':
        clear(args)
    case 'output':
        output(args)
    case 'remove':
        remove(args)
    case 'status':
        status(args)
