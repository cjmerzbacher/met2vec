from unittest import TestCase

from fluxDataset import *
from fluxFile import FluxFile

np = os.path.normpath

#def get_file_name_from_sample_file(file : str):
#    return os.path.basename(file).removesuffix('.csv')
#
#def get_model_name_from_file_name(file : str):
#    """Extracts the common name between sbml model and the sample file."""
#    end_pattern = r'_([0-9]|k)+(\([0-9]*\))?$'
#    return re.sub(end_pattern, '', file)
#
#def get_model_name_from_sample_file(file : str):
#    file_name = get_file_name_from_sample_file(file)
#    model_name = get_model_name_from_file_name(file_name)
#    return model_name
#
#def get_gem_file(model_name : str, main_folder : str):
#    return os.path.join(main_folder, GEM_PATH_FOLDER, f"{model_name}.xml")
#
#def get_model(model_name : str, main_folder : str):
#    gem_file = get_gem_file(model_name, main_folder)
#    try:
#        model = read_sbml_model(gem_file)
#        return model
#    except:
#        return None
#
#def get_model_from_sample_file(sample_file : str, main_folder : str):
#    """Get the cobra model for a geven sample file name"""
#    model_name = get_model_name_from_sample_file(sample_file)
#    return get_model(model_name, main_folder)

class FluxFileTest(TestCase):
    def test_agren_hc_relations(self):
        lat_vent_csv = './lateral_ventricle_-_glial_cells_2_10000.csv'
        lat_vent_file_name = 'lateral_ventricle_-_glial_cells_2_10000'
        lat_vent_model_name = 'lateral_ventricle_-_glial_cells_2'
        lat_vent_gem_file = './gems/lateral_ventricle_-_glial_cells_2.xml'

        ff = FluxFile(lat_vent_csv)

        assert lat_vent_file_name == ff.file_name
        assert lat_vent_model_name == ff.model_name
        assert np(lat_vent_gem_file) == np(ff.gem_file)
