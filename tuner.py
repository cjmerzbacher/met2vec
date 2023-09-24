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
    parser_setup = subparsers.add_parser(SETUP)
    parser_setup.add_argument('-H' ,'--hyperparameter', action='append', nargs=3, dest='hyperparameters')

    # add command
    parser_add = subparsers.add_parser(ADD)
    parser_add.add_argument('name', type=str, help='The argument/name for the hyperparameter.')
    parser_add.add_argument('--type', choices=['str', 'float', 'int'], required=True)
    parser_add.add_argument('--samples', type=int, default=10)
    parser_add.add_argument('--choices', nargs='+', type=str, default='str')
    parser_add.add_argument('--range', nargs=2, type=float)
    parser_add.add_argument('--erange', nargs=2, type=float)

    # run command
    parser_run = subparsers.add_parser(RUN)
    parser_run.add_argument('script', type=str)

    # output command
    parser_output = subparsers.add_parser(OUTPUT)
    parser_output.add_argument('filename') 
    parser_output.add_argument('-a', '--average', type=int, default=10)

    # clear comands
    subparsers.add_parser(CLEAR)

    return parser.parse_args() 

def setup_main_folder(main_folder):
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    
    losses_folder = os.path.join(main_folder, 'losses')
    if not os.path.exists(losses_folder):
        os.mkdir(losses_folder)

    models_folder = os.path.join(main_folder, 'models')
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

def setup_state_file(args):
    state = {'hyperparameters' : {}, 'scripts_run' : 0}
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
        return get_range_values(*hp['erange'], hp['samples'], _type)
    if 'choices' in hp:
        return hp['choices']
    
def get_str_values(values, _type):
    return [f'{v:.4e}' for v in values] if _type == 'float' else [f'{v}' for v in values]

    
def generate_arg_sets(args):
    state = load_state_file(args)
    hyperparameters : dict[str, dict] = state[HPARAMS]

    arg_possabilities = {}
    for name, h in hyperparameters.items():
        values = get_hparam_values(h)
        str_values = get_str_values(values, h['type'])

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

    state['hyperparameters'][args.name] = hyperparameter
    save_state_file(state, args)


def run(args):
    # Generate arg sets
    state = load_state_file(args)
    if ARG_SETS not in state:
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
    save_state_file(state, args)

def output(args):
    state = load_state_file(args)

    output_data = {'run' : [], 'final_loss' : []}
    for i in range(state[SCRIPTS_RUN]):
        file_path = os.path.join(args.main_folder, RUNS, str(i), 'losses.csv')
        losses_df = pd.read_csv(file_path)
        
        losses = losses_df['loss'].values
        average_length = min(len(losses), args.average)
        final_loss = np.mean(losses[-average_length:])

        output_data['run'].append(i)
        output_data['final_loss'].append(final_loss)

        arg_set = state[ARG_SETS][i]
        for name, value in arg_set[ARGS].items():
            if name not in output_data:
                output_data[name] = []
            output_data[name].append(value)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.filename)


    
args = get_args()
print(args)

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

