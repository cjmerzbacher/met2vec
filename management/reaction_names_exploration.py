import argparse
import logging

from cobra.io import read_sbml_model

logging.getLogger('cobra').setLevel(logging.ERROR)

class MetaModel:
    def __init__(self, models=[]):
        self.models = models
        self.metaReactions : dict[str, MetaReaction] = {}

    def add_model(self, path : str):
        model = read_sbml_model(path)
        self.models.append(model)

        for reaction in model.reactions:
            if reaction.id not in self.metaReactions:
                self.metaReactions[reaction.id] = MetaReaction(reaction.id)
            self.metaReactions[reaction.id].add_reaction(reaction)

class MetaReaction:
    def __init__(self, _id):
        self.id = _id
        self.reactions = []
        self.reaction_dicts = []

    def __str__(self):
        return f"<{self.id} {self.is_consistent()} {self.reaction_dicts}>"
        
    def add_reaction(self, reaction):
        self.reactions.append(reaction)
        self.reaction_dicts.append(get_reaction_dict(reaction))

    def is_consistent(self) -> bool:
        return all([self.reaction_dicts[0] == r for r in self.reaction_dicts[1:]])

def read_arguments():
    parser = argparse.ArgumentParser("Flux Renamer", "Basic tool used to rename the columns in a flux sample file to compound names instead of metaids.")
    parser.add_argument("models", metavar="model paths", nargs='+', type=str, help="A list of paths for models to be compared")
    return parser.parse_args()

def get_reaction_dict(reaction):
    M = reaction.metabolites
    return {m.name : M[m] for m in M}

args = read_arguments()

mM = MetaModel()
for model_path in args.models:
    mM.add_model(model_path)

print("\nInconsistent Reactions...")
for mR in mM.metaReactions.values():
    if not mR.is_consistent():
        print(mR)