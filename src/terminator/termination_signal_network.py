from src.terminator.networks import CNN
from scipy.linalg import toeplitz
import numpy as np

from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()


class TerminationSignalNetwork(nn.Module):
    def __init__(self, window, n_ensemble, obs_shape, model_config):
        nn.Module.__init__(self)

        self.window = window
        self.n_ensemble = n_ensemble
        self.bias = nn.Parameter(torch.empty(n_ensemble))
        self.cost_network = CNN(obs_shape=obs_shape,
                                num_outputs=n_ensemble,
                                model_config=model_config,
                                final_activation="relu")

    def forward(self, obs, output_mask=None):
        # obs input shape = [B, N, C, W, H]
        shape = obs.shape

        x = obs.view(-1, *shape[2:])
        x = self.cost_network.forward(x)
        x = x.view(shape[0], shape[1], self.n_ensemble)

        # assumes the full rollout was inputted. Will ouput all window sums
        if shape[1] <= self.window:
            M_numpy = toeplitz(np.ones(shape[1], dtype='float32'), np.zeros(shape[1], dtype='float32'))
        else:
            M_numpy = toeplitz(np.concatenate([np.ones(self.window), np.zeros(shape[1] - self.window)], dtype='float32'),
                               np.zeros(shape[1], dtype='float32'))
        M = torch.tensor(M_numpy).to(x.device)
        x = torch.einsum('jl, ijk -> ilk', M, x)
        x = x.reshape(-1, self.n_ensemble)  # can't use view since x is not contiguous in memory

        x = torch.sigmoid(x - self.bias)
        if output_mask is None:
            return x
        return x.gather(1, output_mask.unsqueeze(-1)).squeeze(-1)
