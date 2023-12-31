from unittest import TestCase
from fluxDataset import FluxDataset

import numpy as np

class FluxDatasetTest(TestCase):
    def init(self):
        self.inited = True

    def test_dataset_loads_small_samples(self):
        self.fd1 = FluxDataset('./data/samples/small_human/', 300, 0, seed=0)
        self.fd2 = FluxDataset('./data/samples/small_human/', 300, 0, seed=0)

        print(self.fd1.renaming_dicts)
        assert self.fd1.renaming_dicts != {}

        assert np.isclose(np.max(self.fd1.normalized_values - self.fd2.normalized_values), 0)

    def test_dataset_multiple_same_renaming(self):
        pass