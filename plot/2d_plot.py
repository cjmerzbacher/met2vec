import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

from misc.fluxDataset import load_fd, get_data
from misc.plot import get_save_parser
from misc.parsing import fluxDataset_loading_parser, make_save_parser

import pandas as pd
import argparse

parser = argparse.ArgumentParser(parents=[
    fluxDataset_loading_parser(),
    get_save_parser(),
])


