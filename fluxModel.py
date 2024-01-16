import logging

from misc.io import *
from misc.constants import GEM_FOLDER, PKL_FOLDER

from cobra.io import read_sbml_model

logging.getLogger('cobra').setLevel(logging.CRITICAL)




def get_reaction_name(reaction):
    reaction_parts = [f'{m.name}({m.compartment})[{reaction.metabolites[m]}]' for m in reaction.metabolites]
    name = "".join(sorted(reaction_parts))
    return name.replace(",", ".")

def get_rename_dict(model) -> dict[str,str]:
    """Gets the renaming dict for a given file.
    
    This will transform the file path to get the name and path for the reactions.
    Then the sbml model loaded will be used to generate a dictionary mapping metabolite
    names to 'reaction_names'.
    
    Arguments:
        path: The path of the sample which will be used to find a model.

    Returns:
        renaming: A dictionary mapping metabolits names to 'reaction_names'
    """
    if model == None:
        return None
    return {r.id : get_reaction_name(r) for r in model.reactions}

class FluxModel:
    def __init__(self, name, main_folder):
        self.name = name
        self.main_folder = main_folder

        self.gem_folder = join(main_folder, GEM_FOLDER, raise_exists=True)
        self.path = join(self.gem_folder, f"{name}.xml", raise_exists=True)

        self.pkl_folder = join(self.gem_folder, PKL_FOLDER, make_folder=True)
        self.pkl_path = join(self.pkl_folder, f"{name}.pkl")
        self.join_path = join(self.pkl_folder, f"{name}_join.json")

    def make_cobra_model_pkl(self):
        cobra_model = read_sbml_model(self.path)
        safe_pkl_dump(self.pkl_path, cobra_model)
        return cobra_model

    def get_cobra_model(self):
        cobra_model = safe_pkl_load(self.pkl_path)
        if cobra_model is None:
            cobra_model = self.make_cobra_model_pkl()
        return cobra_model

    def make_renaming_dict(self):
        cobra_model = self.get_cobra_model()
        renaming_dict = get_rename_dict(cobra_model)
        safe_json_dump(self.join_path, renaming_dict) 
        return renaming_dict

    def get_renaming_dict(self):
        renaming_dict = safe_json_load(self.join_path)
        if renaming_dict is None:
            renaming_dict = self.make_renaming_dict()
        return renaming_dict


