from unittest import TestCase
from fluxDataset import FluxDataset

class FluxDatasetTest(TestCase):
    def test_dataset_loads_small_samples(self):
        fd = FluxDataset('./data/samples/small_human/')
        print(fd.renaming_dicts)
        
        assert fd.renaming_dicts != {}