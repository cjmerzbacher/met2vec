from unittest import TestCase

from misc.constants import *
from fluxModel import *

np = os.path.normpath

liver_hep = "liver_hepatocytes"
small_human_folder = "./data/samples/small_human/"


class FluxModelTest(TestCase):
    def test_small_human_model_paths(self):
        model_path = "./data/samples/small_human/gems/liver_hepatocytes.xml"

        fm = FluxModel(liver_hep, small_human_folder)

        assert fm.name == liver_hep
        assert fm.main_folder == small_human_folder
        assert fm.gem_folder == os.path.join(small_human_folder, GEM_FOLDER)
        assert fm.pkl_folder == os.path.join(fm.gem_folder, PKL_FOLDER)
        assert np(fm.path) == np(model_path)

    def test_small_human_model_renaming(self):
        fm = FluxModel(liver_hep, small_human_folder)

        model_renaming_dict_1 = fm.get_renaming_dict()
        fm.make_cobra_model_pkl()
        model_renaming_dict_2 = fm.get_renaming_dict()

        for reaction in model_renaming_dict_1:
            assert model_renaming_dict_1[reaction] == model_renaming_dict_2[reaction]

