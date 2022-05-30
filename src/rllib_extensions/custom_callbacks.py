from typing import Dict, Optional
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from src.terminator.terminator import Terminator


class CustomCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        DefaultCallbacks.__init__(self, legacy_callbacks_dict)

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        self.keys = ['aggregated_cost', 'dead', 'cost_err', 'real_reward']
        for key in self.keys:
            episode.user_data[key] = []
            episode.hist_data[key] = []
        episode.user_data["real_tot_reward"] = []
        episode.hist_data["real_tot_reward"] = []

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        info = episode.last_info_for()
        for key in self.keys:
            if key in info:
                episode.user_data[key].append(info[key])
            else:
                episode.user_data[key].append(0)

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
    ) -> None:
        # assert episode.batch_builder.policy_collectors["default_policy"].batches[-1][
        #     "dones"
        # ][-1], (
        #     "ERROR: `on_episode_end()` should only be called " "after episode is done!"
        # )
        for key in self.keys:
            mean_val = np.mean(episode.user_data[key])
            episode.custom_metrics[key] = mean_val
            episode.hist_data[key] = episode.user_data[key]
        info = episode.last_info_for()
        if 'episode' in info:
            episode.custom_metrics["real_tot_reward"] = info['episode']['r']


    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: Episode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        episode.custom_metrics["terminator_loss"] = 0

    def on_train_result(self, *, trainer: "Trainer", result: dict,
                        **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer: Current trainer instance.
            result: Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        if trainer.config["learn_costs"]:
            result['custom_metrics']['terminator_loss_mean'] = trainer.workers.local_worker().terminator_info['terminator_loss']

