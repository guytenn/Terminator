import argparse
import copy
import multiprocessing

import glob
import os
import datetime
import time
import numpy as np
import ray as ray
from ray.rllib.agents.ppo import PPOTrainer
from src.rllib_extensions.ppo_trainer import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from src.rllib_extensions.custom_callbacks import CustomCallbacks
from src.terminator.complex_input_net import ComplexInputNetwork
from src.envs.backseat_driver.backseat_driver import BackseatDriver
from src.envs.base_env import CostMethod

import wandb


def main(args, wandb_logger):
    trainer_cls = PPOTrainer

    config = copy.deepcopy(trainer_cls.get_default_config())

    if args.learn_costs:
        cost_method = CostMethod.LEARNED
    elif args.cost_in_state:
        cost_method = CostMethod.REAL
    else:
        cost_method = CostMethod.NONE

    config["env_config"] = dict(cost_method=cost_method,
                                cost_in_state=args.cost_in_state or args.learn_costs,
                                cost_history_in_state=args.cost_history_in_state,
                                cost_coef=args.cost_coef,
                                window=args.env_window,
                                no_termination=args.no_termination,
                                termination_penalty=args.termination_penalty,
                                env_path='src/envs/backseat_driver/build/')

    config["learn_costs"] = args.learn_costs
    config["framework"] = "torch"
    config["log_level"] = "ERROR" # "DEBUG"
    config["horizon"] = 10000
    config["no_done_at_end"] = True

    filters_42x42 = [
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]

    ModelCatalog.register_custom_model("complex_input", ComplexInputNetwork)
    config["model"] = {
                        "custom_model": "complex_input",
                        "conv_filters": filters_42x42,
                        "post_fcnet_hiddens": [400, 256],
                        "use_lstm": args.use_lstm,
                        "lstm_cell_size": 256,
                       }

    config['observation_space'] = BackseatDriver(**config["env_config"]).observation_space

    # i = 0
    # while os.path.exists(os.path.join(f'src/data/cost_net_{i}.pt')):
    #     i += 1
    cost_model_fname = f'src/data/cost_net_{int(time.time() * 1000)}.pt'

    terminator_config = dict(learning_rate=float(1e-3),
                             batch_size=args.term_batch_size,
                             n_ensemble=args.n_ensemble,
                             replay_size=args.term_replay_size,
                             window=args.window,
                             cost_history_in_state=args.cost_history_in_state,
                             train_steps=args.term_train_steps,
                             cost_model_fname=cost_model_fname,
                             bonus_type=args.bonus_type,
                             bonus_coef=args.bonus_coef,
                             reward_penalty_coef=args.reward_penalty_coef,
                             reward_bonus_coef=args.reward_bonus_coef)

    config['terminator_config'] = terminator_config

    config["rollout_fragment_length"] = 50
    if args.debug:
        config["num_workers"] = 0
    else:
        config["num_workers"] = args.num_processes
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    config["num_gpus"] = args.num_gpus
    config["num_cpus_per_worker"] = 0
    config["entropy_coeff"] = args.entropy_coeff
    if args.debug:
        config['train_batch_size'] = 1000
        config['num_sgd_iter'] = args.num_epochs
        config['sgd_minibatch_size'] = 32
    else:
        config['train_batch_size'] = args.train_batch_size
        config['num_sgd_iter'] = args.num_epochs
        config['sgd_minibatch_size'] = args.batch_size

    config["seed"] = args.seed

    config["callbacks"] = CustomCallbacks
    if args.learn_costs and args.termination_gamma:
        config["gamma"] = "termination"
    else:
        config["gamma"] = 0.99

    config["env_config"].update({"model_config": config["model"], "terminator_config": terminator_config})

    def env_creator(env_config):
       env = BackseatDriver(**env_config)
       return env

    register_env("backseat_driver-v0", env_creator)

    trainer = trainer_cls(env="backseat_driver-v0", config=config)

    t = 0
    print('-' * 20)
    print('Training')

    while t < args.train_timesteps:
        try:
            result = trainer.train()
        except Exception as e:
            print(F'ERROR: {e}')
            continue

        t = result['timesteps_total']
        if args.learn_costs:
            terminator_loss = result['custom_metrics']['terminator_loss_mean']
            if terminator_loss < 0:
                terminator_loss = "initializing"
            else:
                terminator_loss = f"{terminator_loss:.2f}"
            cost_err = f"{result['custom_metrics']['cost_err_mean']:.2f}"
        else:
            terminator_loss = "disabled"
            cost_err = "disabled"
        print(
          f"Iteration: {result['training_iteration']}, "
          f"total timesteps: {result['timesteps_total']}, "
          f"total time: {result['time_total_s']:.1f}, "
          f"FPS: {result['timesteps_total'] / result['time_total_s']:.1f}, "
          f"real reward: {result['custom_metrics']['real_reward_mean']:.1f}, "
          f"real tot reward: {result['custom_metrics']['real_tot_reward_mean']:.1f}, "
          f"mean reward: {result['episode_reward_mean']:.1f}, "
          f"min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}, "
          f"mean_agg_cost: {result['custom_metrics']['aggregated_cost_mean']:.1f}, "
          f"mean_dead: {result['custom_metrics']['dead_mean']:.1f}, "
          f"terminator_loss: {terminator_loss}, "
          f"cost_err: {cost_err}, "
          f"entropy: {result['info']['learner']['default_policy']['learner_stats']['entropy']:.1f}, "
          f"policy loss: {result['info']['learner']['default_policy']['learner_stats']['policy_loss']:.1f}")
        print('--' * 20)

        if args.wandb:
            results_to_log = dict(real_reward=result['custom_metrics']['real_reward_mean'],
                                  real_tot_reward=result['custom_metrics']['real_tot_reward_mean'],
                                  mean_reward=result['episode_reward_mean'],
                                  min_reward=result['episode_reward_min'],
                                  max_reward=result['episode_reward_max'],
                                  aggregated_cost=result['custom_metrics']['aggregated_cost_mean'],
                                  dead=result['custom_metrics']['dead_mean'])
            if args.learn_costs and terminator_loss != "initializing":
                results_to_log.update(dict(terminator_loss=result['custom_metrics']['terminator_loss_mean'],
                                           cost_err=result['custom_metrics']['cost_err_mean']))
            wandb_logger.log(results_to_log, step=result['training_iteration'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Terminator')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--train-batch-size', type=int, default=1024,
                        help='Number of timesteps collected for each SGD round. '
                             'This defines the size of each SGD epoch. (default: 1024)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch. (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch). (default: 5)')
    parser.add_argument('--graphics', action='store_true', default=False,
                        help='When enabled will render environment')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Log to wandb')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Log to wandb')
    parser.add_argument('--project-name', default='Terminator',
                        help='Project name for wandb logging')
    parser.add_argument('--run-name', default='',
                        help='run name for wandb logging')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of workers during training (default = -1, use all cpus)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of gpus (default = 1)')
    parser.add_argument('--entropy-coeff', type=float, default=0.0,
                        help='Entropy loss coefficient (default = 0.01)')
    parser.add_argument('--cost_coef', type=float, default=1,
                        help='Cost coefficient for termination in environment (default = 1)')
    parser.add_argument('--bonus_coef', type=float, default=1,
                        help='Bonus coefficient for termination cost confidence in TermPG (default = 1)')
    parser.add_argument('--bonus_type', default='maxmin', choices=['none', 'std', 'maxmin'],
                        help='Type of bonus to use for costs')
    parser.add_argument('--reward_penalty_coef', type=float, default=0,
                        help='Penalty coefficient for costs')
    parser.add_argument('--termination_penalty', type=float, default=0,
                        help='A Penalty for termination')
    parser.add_argument('--reward_bonus_coef', type=float, default=0,
                        help='Bonus coefficient for optimism in costs')
    parser.add_argument('--window', type=int, default=30,
                        help='Window size for termination.')
    parser.add_argument('--env_window', type=int, default=-1,
                        help='The real window the env will use for termination. If -1 will use default window.')
    parser.add_argument('--n_ensemble', type=int, default=3,
                        help='Number of networks to use in cost model ensemble.')
    parser.add_argument('--term_train_steps', type=int, default=30,
                        help='Number of train steps to train terminator.')
    parser.add_argument('--term_batch_size', type=int, default=64,
                        help='Batch size for terminator.')
    parser.add_argument('--term_replay_size', type=int, default=1000,
                        help='Replay size for terminator.')
    parser.add_argument('--clean_data', action='store_true', default=False,
                        help='Will remove all model files in src/data')
    parser.add_argument('--use_lstm', action='store_true', default=False)
    parser.add_argument('--cost_in_state', action='store_true', default=False)
    parser.add_argument('--no_termination', action='store_true', default=False)
    parser.add_argument('--cost_history_in_state', action='store_true', default=False)
    parser.add_argument('--learn_costs', action='store_true', default=False)
    parser.add_argument('--termination_gamma', action='store_true', default=False)

    args = parser.parse_args()

    if args.clean_data:
        files = glob.glob('src/data/cost_net*')
        for f in files:
            os.remove(f)

    args.seed = np.random.randint(2 ** 30 - 1)

    if args.env_window == -1:
        args.env_window = args.window

    if args.num_processes == -1:
        args.num_processes = None

    if args.wandb:
        wandb.login()
        wandb_logger = wandb.init(project=args.project_name, name=args.run_name, config=args.__dict__)
    else:
        wandb_logger = None

    if args.num_processes is None:
        args.num_processes = multiprocessing.cpu_count()

    ray.init(num_cpus=args.num_processes)

    main(args, wandb_logger)


