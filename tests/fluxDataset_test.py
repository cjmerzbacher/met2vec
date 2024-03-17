from unittest import TestCase
from fluxDataset import FluxDataset

import numpy as np
from misc.constants import *

class FluxDatasetTest(TestCase):
    def test_dataset_loads_small_samples(self):
        seed = 0

        fd1 = FluxDataset('./data/samples/small_human/', 300, seed=seed)

        assert fd1.seed == seed

        # Check Data Integrity
        assert np.allclose(fd1.data.drop(columns=SOURCE_COLUMNS) @ fd1.S.values.T, 0)

        fd2 = FluxDataset('./data/samples/small_human/', 300, seed=seed)

        for flux_file in fd1.flux_files.values():
            assert flux_file.seed == fd1.seed

        assert np.allclose(fd1.values, fd2.values)

