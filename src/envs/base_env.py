import numpy as np
from gym import core, spaces
from collections import deque
import torch
from enum import Enum
from src.terminator.networks import CNN


class CostMethod(Enum):
    REAL = 1
    LEARNED = 2
    NONE = 3


class BaseEnv(core.Env):
    def __init__(self, observation_shape, n_actions,
                 cost_coef=1., cost_in_state=True, cost_bias=6, window=5,
                 cost_history_in_state=False, cost_method=CostMethod.REAL,
                 model_config=None, terminator_config=None,
                 no_termination=False, termination_penalty=0,
                 **kwargs):

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)
        self.model_config = model_config
        self.terminator_config = terminator_config
        self.no_termination = no_termination
        self.termination_penalty = termination_penalty

        self.cost_method = cost_method
        self.cost_in_state = cost_in_state
        self.cost_history_in_state = cost_history_in_state
        self.cost_coef = cost_coef
        self.cost_bias = cost_bias
        self.window = window

        self.aggregated_cost = 0
        self.cost_memory = deque([0] * window)
        self.est_aggregated_cost = 0
        self.est_cost_memory = deque([0] * window)

        self.nsteps = 0
        self.tot_reward = 0

        self.rng = None
        self.seed()

        self.learned_cost_model = None
        self.cost_model_fname = None
        if terminator_config is not None and cost_method == CostMethod.LEARNED:
            self.cost_model_fname = terminator_config["cost_model_fname"]
            self.learned_cost_model = self.init_cost_net()
        self.cost_model = None

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def after_reset(self):
        self.aggregated_cost = 0
        self.cost_memory = deque([0] * self.window)
        self.est_aggregated_cost = 0
        self.est_cost_memory = deque([0] * self.window)
        self.nsteps = 0
        self.tot_reward = 0

    def after_step(self, s, r, d, i, c):
        if not self.no_termination:
            if self.cost_model is not None:
                c, _ = self._estimate_cost(s, self.cost_model, 'maxmin')
            self.cost_memory.append(c)
            self.aggregated_cost += self.cost_coef * (c - self.cost_memory.popleft())
            die_prob = 1 / (1 + np.exp(-(self.aggregated_cost - self.cost_bias)))
            dead = self.rng.rand() <= die_prob
        else:
            dead = d

        done = d or dead

        estimated_cost = 0
        cost_bonus = 0
        if self.cost_method == CostMethod.NONE:
            estimated_cost = 0
        elif self.cost_method == CostMethod.LEARNED:
            if self.learned_cost_model is not None:
                estimated_cost, cost_bonus = self._estimate_cost(s, self.learned_cost_model, self.terminator_config["bonus_type"])
                i.update({"cost_err": np.abs(c - estimated_cost)})
            else:
                estimated_cost = 0
        # else c remains unchanged

        self.est_cost_memory.append(estimated_cost)
        self.est_aggregated_cost += estimated_cost - self.est_cost_memory.popleft()

        if self.cost_in_state:
            if self.cost_method == CostMethod.LEARNED:
                est_aggregated_cost, est_cost_memory = self.est_aggregated_cost, self.est_cost_memory
            elif self.cost_method == CostMethod.REAL:
                est_aggregated_cost, est_cost_memory = self.aggregated_cost, self.cost_memory
            else:
                raise ValueError("Cost Method must be either REAL or LEARNED when cost_in_state == true")
            s[:, :, -1] = est_aggregated_cost
            if self.cost_history_in_state:
                base_shape = s.shape[0]
                s[:, :, -self.window - 1:-1] = np.repeat(est_cost_memory, base_shape ** 2). \
                    reshape((-1, base_shape, base_shape)).swapaxes(0, 2)

        self.tot_reward += r
        self.nsteps += 1

        i.update({'dead': dead, 'aggregated_cost': self.aggregated_cost, 'real_reward': r})
        if done:
            i['episode'] = {'r': self.tot_reward, 'l': self.nsteps}

        if self.terminator_config is not None:
            if self.terminator_config["reward_penalty_coef"] > 0:
                r -= self.terminator_config["reward_penalty_coef"] * self.est_aggregated_cost
            if self.terminator_config["reward_bonus_coef"] > 0:
                r += self.terminator_config["reward_bonus_coef"] * cost_bonus

        if dead and self.termination_penalty > 0:
            r -= self.termination_penalty

        return s, r, done, i

    def seed(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

    def set_cost_net(self):
        try:
            state_dict = torch.load(self.cost_model_fname, map_location="cpu")
            self.learned_cost_model.load_state_dict(state_dict)
        except:
            print("Couldn't load model")

    def init_cost_net(self):
        return CNN(obs_shape=(*self.observation_space.shape[0:2],
                              self.observation_space.shape[2] - 1 - self.window * self.cost_history_in_state),
                   num_outputs=self.terminator_config["n_ensemble"],
                   model_config=self.model_config,
                   final_activation="relu")

    def _estimate_cost(self, s, model, bonus_type):
        costs = model(torch.tensor(s[np.newaxis, :, :, :])[:, :, :, :-1 - self.cost_in_state * self.cost_history_in_state * self.window])
        if bonus_type == 'maxmin':
            max_costs = costs.max().item()
            min_costs = costs.min().item()
            return costs.min().item(), max_costs - min_costs
        elif bonus_type == 'std':
            cost_std = costs.std().item()
            return costs.mean().item() - self.terminator_config["bonus_coef"] * cost_std, cost_std
        else:  # bonus_type == none
            return costs.mean().item(), 0.