# Terminator (TermPG)
#### Open-source codebase for Terminator Policy Gradient (TermPG), from ["Reinforcement Learning with a Terminator"](https://url).

### Installation

To use Terminator, make sure python3 is installed and pip is up to date. This project was tested using python verison 3.8.
#### Clone the Repository
```bash
    git clone https://github.com/guytenn/Terminator.git
```
It is recommended to install requirements using a virtual environment. To set up a virtual environment this follow this steps
```bash
    cd Terminator/
    python3 -m venv terminator_env
    source terminator_env/bin/activate
    # Upgrade Pip
    pip install --upgrade pip
```

#### Install Requirements
While in Terminator directory install requirements using the following command
```bash 
    pip install .
```

#### Download Backseat Driver
You can find the latest version of Backseat Driver [here](https://drive.google.com/file/d/10yn3F84ahvJORED5TVaicVF6Ff-UdYfE/view?usp=sharing) (currently only supports linux).

Download and unzip the files to src/envs/backseat_driver/build/

Your file system should be organized as follows.
```bash 
    src/envs/backseat_driver/build/BackseatDriverTerm_BurstDebugInformation_DoNotShip
    src/envs/backseat_driver/build/BackseatDriverTerm_Data
    src/envs/backseat_driver/build/BackseatDriverTerm.x86_64
    src/envs/backseat_driver/build/UnityPlayer.so
```

## Usage
### Quick start

The following will run TermPG using its default parameters
```bash
    python3 run.py --learn_costs --termination_gamma
```
Below you can find a list of arguments you can change

| TermPG Arguments              | Description|
|:------------------------------|:-------------|
| `--learn_costs`               |Learn costs according to TermPG
| `--termination_gamma`         |Use dynamic discount factor according to TermPG
| `--cost_coef 1`               |Cost coefficient for termination in environment
| `--bonus_coef 1`              |Bonus coefficient for termination cost confidence in TermPG
| `--bonus_type 'maxmin'`       |Type of bonus to use for costs (one of: 'none', 'std', 'maxmin')
| `--reward_penalty_coef 0`     |Penalty coefficient for costs (penalize reward by estimated costs)
| `--termination_penalty 0`     |A Penalty for termination (reward shaping variant)
| `--reward_bonus_coef 0`       |Bonus coefficient for optimism in costs
| `--window 30`                 |Window size for termination
| `--env_window -1`             |The real window the env will use for termination. If -1 will use default window.
| `--n_ensemble 3`              |Number of networks to use in cost model ensemble
| `--term_train_steps 30`       |Number of train steps to train terminator
| `--term_batch_size 64`        |Batch size for terminator
| `--term_replay_size 1000`     |Replay size for terminator
| `--cost_in_state`             |Will add true cost to state (TerMDP with *known* costs)
| `--no_termination`            |Will disable termination in environment
| `--cost_history_in_state`     |Will add history of costs to state in addition to accumulated cost

| General Arguments           | Description|
|:----------------------------|:-------------|
| `--train_timesteps 1000000` |Number of simulation timesteps to train a policy
| `--train_batch_size 1024`   |Number of timesteps collected for each SGD round. This defines the size of each SGD epoch. 
| `--batch_size 32`           |Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
| `--num_epochs 3`            |Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
| `--graphics`                |When enabled will render environment
| `--wandb`                   |Log to wandb
| `--project_name`            |Project name for wandb logging
| `--run_name`                |Run name for wandb logging
| `--num_processes 8`         |Number of workers during training (value of -1 will use all cpus)
| `--num_gpus 1`              |Number of gpus to use for training
| `--entropy_coeff 0`         |Entropy loss coefficient 
| `--use_lstm`                |Use a recurrent policy
| `--clean_data`              |Will remove all model files in src/data
 
[//]: # (## Citation)

[//]: # (To cite our paper please use)

[//]: # ()
[//]: # (```)

[//]: # (@Article{young19minatar,)

[//]: # (author = {{Young}, Kenny and {Tian}, Tian},)

[//]: # (title = {MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments},)

[//]: # (journal = {arXiv preprint arXiv:1903.03176},)

[//]: # (year = "2019")

[//]: # (})

[//]: # (```)