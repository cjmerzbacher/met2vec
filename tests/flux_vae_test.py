from unittest import TestCase

from fluxDataset import *
from vaeTrainer import *
from vae import FluxVAE

from misc.constants import *

import numpy as np

import torch

class FluxVAETest(TestCase):
    def test_loss_properties(self):
        fd = FluxDataset("./data/samples/small_human")

        v = fd.normalized_values
        S = fd.S.values.T
        v_mu = fd.flux_mean.values
        v_std = fd.flux_std.values

        assert np.allclose(((v * v_std) + v_mu) @ S, 0)
        
        vae = FluxVAE(
            n_in = len(fd.core_reaction_names),
            n_emb = 16,
            n_lay = 3,
            lrelu_slope=0.1,
            batch_norm=False,
            dropout_p=0,
            reaction_names=fd.core_reaction_names
        )

        loss, blame = vae.loss(
            v=v, 
            v_r=v, 
            mu=torch.zeros(16), 
            log_var=torch.ones(16),
            S=S,
            v_mu=v_mu,
            v_std=v_std,
            beta_S=1  
        ) 

        print(blame)

        assert blame[S_LOSS] < 1e-3