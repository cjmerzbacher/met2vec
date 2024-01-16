from metaReaction import MetaReaction
from cobra.io import read_sbml_model

import logging
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