from unittest import TestCase

from fluxDataset import *

np = os.path.normpath

class FluxDatasetTest(TestCase):
    def test_agren_hc_relations(self):
        lat_vent_csv = 'lateral_ventricle_-_glial_cells_2_10000.csv'
        lat_vent_file_name = 'lateral_ventricle_-_glial_cells_2_10000'
        lat_vent_model_name = 'lateral_ventricle_-_glial_cells_2'
        lat_vent_xml = './gems/lateral_ventricle_-_glial_cells_2.xml'

        file_name = get_file_name_from_sample_file(lat_vent_csv)
        model_name = get_model_name_from_file_name(file_name)
        gem_file = get_gem_file(model_name, "./")

        assert file_name == lat_vent_file_name
        assert model_name  == lat_vent_model_name
        assert np(gem_file) == np(lat_vent_xml)
