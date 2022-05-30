from src.terminator.termination_signal_network import TerminationSignalNetwork
import numpy as np
import torch
import torch.nn as nn
import GPUtil
from collections import deque
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch
from src.rllib_extensions.custom_postprocessing import compute_gae_for_sample_batch
import os


class Terminator:
    def __init__(self, workers, obs_space, model_config, terminator_config, single_gpu=True):
        self.workers = workers
        self.window = terminator_config['window']
        self.cost_history_in_state = terminator_config['cost_history_in_state']
        self.input_shape = (*obs_space.shape[0:2], obs_space.shape[2] - 1 - self.window * self.cost_history_in_state)

        if torch.cuda.is_available():
            if single_gpu:
                self.device = torch.device(f'cuda:0')
            else:
                deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
                self.device = torch.device(f'cuda:{deviceIds[0]}')

        else:
            self.device = torch.device('cpu')

        self.termination_model = TerminationSignalNetwork(terminator_config['window'],
                                                          terminator_config['n_ensemble'],
                                                          self.input_shape,
                                                          model_config).to(self.device)

        self.n_ensemble = terminator_config['n_ensemble']

        self.replay_size = terminator_config['replay_size']
        self.replay = dict(rollouts=deque([]), labels=deque([]))
        self.curr_replay_size = 0
        self.last_idx = 0

        self.n_train_steps = terminator_config["train_steps"]

        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.termination_model.parameters(),
                                          lr=float(terminator_config['learning_rate']))

        self.cost_model_fname = terminator_config["cost_model_fname"]

    def __call__(self, samples):
        rollouts = samples.split_by_episode()
        for i in range(len(rollouts)):
            clean_states = rollouts[i][SampleBatch.OBS][:, :, :, :self.input_shape[-1]]  # state excluding cost channel
            self.replay['rollouts'].append(clean_states)
            self.replay['labels'].append(rollouts[i]['infos'][-1]['dead'])
            if len(self.replay['labels']) > self.replay_size:
                self.replay['rollouts'].popleft()
                self.replay['labels'].popleft()

        for _ in range(self.n_train_steps):
            terminator_loss = self.train_step()

        torch.save(self.termination_model.cost_network.state_dict(), self.cost_model_fname)
        self.workers.foreach_env(lambda env: env.set_cost_net())
        self.workers.local_worker().terminator_info = dict(terminator_loss=terminator_loss)

        return samples

    def train_step(self):
        batch_idx = np.random.choice(range(len(self.replay['labels'])), self.n_ensemble)
        lens = [len(self.replay['rollouts'][batch_idx[i]]) for i in range(self.n_ensemble)]
        padded_inputs = torch.tensor(np.stack([np.pad(self.replay['rollouts'][batch_idx[i]],
                                                      ((0, max(lens) - len(self.replay['rollouts'][batch_idx[i]])),
                                                       (0, 0), (0, 0), (0, 0)), constant_values=0)
                                               for i in range(self.n_ensemble)])).to(self.device)
        output_mask = torch.concat([i * torch.ones(max(lens)) for i in range(self.n_ensemble)]).long().to(self.device)
        res = self.termination_model.forward(padded_inputs, output_mask=output_mask)

        mask = torch.concat([torch.concat([torch.ones(lens[i]), torch.zeros(max(lens) - lens[i])]) for i in range(self.n_ensemble)]).to(self.device)
        labels = torch.tensor(
            np.concatenate([np.zeros((max(lens)-1, self.n_ensemble)),
                            np.array([self.replay['labels'][batch_idx[i]]
                                      for i in range(self.n_ensemble)])[np.newaxis]], axis=0, dtype="float32")
                .reshape(-1, order='F')).to(self.device)

        self.optimizer.zero_grad()
        loss = self.bce_loss(res * mask, labels)
        loss.backward()
        self.optimizer.step()
        tot_loss = loss.item()

        return tot_loss / self.n_ensemble


