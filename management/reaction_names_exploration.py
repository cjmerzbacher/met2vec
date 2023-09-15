import argparse
from metaModel import MetaModel

def read_arguments():
    parser = argparse.ArgumentParser("Flux Renamer", "Basic tool used to rename the columns in a flux sample file to compound names instead of metaids.")
    parser.add_argument("models", metavar="model paths", nargs='+', type=str, help="A list of paths for models to be compared")
    return parser.parse_args()

args = read_arguments()

mM = MetaModel()
for model_path in args.models:
    mM.add_model(model_path)

print("\nInconsistent Reactions...")
for mR in mM.metaReactions.values():
    if not mR.is_consistent():
        print(mR)