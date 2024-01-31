import logging
import pandas as pd

from misc.io import *
from misc.constants import GEM_FOLDER, PKL_FOLDER

from cobra import Metabolite, Reaction, Model
from cobra.io import read_sbml_model


logging.getLogger('cobra').setLevel(logging.CRITICAL)

def get_metabolite_name(metabolite : Metabolite):
    return f"m[f[{metabolite.formula}]ch[{metabolite.charge}]co[{metabolite.compartment}]]"

def get_reaction_name(reaction):
    reaction_parts = [f"{get_metabolite_name(m)}[{reaction.metabolites[m]}]" for m in reaction.metabolites]
    name = f"r[{''.join(sorted(reaction_parts))}]"
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

def get_reaction_stoicheometry(reaction : Reaction):
    stoicheometry = {
        get_metabolite_name(m) : reaction.metabolites[m]
        for m in reaction.metabolites
    }
    return stoicheometry

def get_model_stoicheometry(model : Model):
    data = {
        get_reaction_name(r) :
        get_reaction_stoicheometry(r)
        for r in model.reactions
    }
    return pd.DataFrame(data).fillna(0)

class FluxModel:
    def __init__(self, name, main_folder):
        self.name = name
        self.main_folder = main_folder

        self.gem_folder = join(main_folder, GEM_FOLDER, raise_exists=True)
        self.path = join(self.gem_folder, f"{name}.xml", raise_exists=True)

        self.pkl_folder = join(self.gem_folder, PKL_FOLDER, make_folder=True)
        self.pkl_path = join(self.pkl_folder, f"{name}.pkl")
        self.join_path = join(self.pkl_folder, f"{name}_join.json")

    def make_cobra_model_pkl(self) -> Model:
        cobra_model = read_sbml_model(self.path)
        safe_pkl_dump(self.pkl_path, cobra_model)
        return cobra_model

    def get_cobra_model(self) -> Model:
        cobra_model = safe_pkl_load(self.pkl_path)
        if cobra_model is None:
            cobra_model = self.make_cobra_model_pkl()
        return cobra_model
    
    def get_stoicheometry(self) -> pd.DataFrame:
        return get_model_stoicheometry(self.get_cobra_model())

    def get_renaming_dict(self) -> dict[str,str]:
        cobra_model = self.get_cobra_model()
        renaming_dict = get_rename_dict(cobra_model)
        return renaming_dict


