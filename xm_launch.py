r"""Launches a2perf experiments on xcloud.

Due to environment variable compatibility issues with zsh, this command should be executed using a 'bash EOF' approach. Provide the command inputs as follows:

Example usages:

bash <<EOF
xmanager_dev launch launch/xcloud_launch.py -- --xm_resource_alloc=group:xcloud/xcloud-shared-user \
  --xm_gcs_path=gs://xcloud-shared/ikechukwuu/a2perf/ \
  --noxm_monitor_on_launch \
  --experiment_name="A2Perf. Quadruped Locomotion. PPO " \
  --domain=quadruped_locomotion \
  --algos=ppo \
  --tasks=dog_pace,dog_trot,dog_spin \
  --seeds=37,68,24 \
  --debug

EOF

bash <<EOF
xmanager_dev launch launch/xcloud_launch.py -- --xm_resource_alloc=group:xcloud/xcloud-shared-user \
  --xm_gcs_path=gs://xcloud-shared/ikechukwuu/a2perf/ \
  --noxm_monitor_on_launch \
  --experiment_name="A2Perf. Quadruped Locomotion. SAC " \
  --domain=quadruped_locomotion \
  --algos=sac \
  --tasks=dog_pace,dog_trot,dog_spin \
  --seeds=37,68,24 \
  --debug
EOF

This command initiates the A2Perf experiments on the xcloud platform, specifying necessary parameters such as resource allocation, GCS path, experiment name, algorithms, and tasks.
"""

import itertools
import os

from absl import app
from absl import flags
from absl import logging
from xmanager import xm
from xmanager import xm_local

_ALGOS = flags.DEFINE_list(
    'algos',
    ['ppo'],
    'Algorithms to run. If multiple are specified, they will be run in'
    ' sequence',
)
_VOCABULARY_MANAGER_AUTH_KEY = flags.DEFINE_string(
    'vocabulary_manager_auth_key',
    '',
    'Authentication key for the manager server.',
)
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to use')

_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
_DOMAIN = flags.DEFINE_enum(
    'domain',
    'quadruped_locomotion',
    ['quadruped_locomotion', 'web_navigation', 'circuit_training'],
    'Domain to run',
)
_DATASETS_PATH = flags.DEFINE_string(
    'datasets_path',
    None,
    'Path for Minari datasets',
)
_POLICY_NAME = flags.DEFINE_string(
    'policy_name',
    'policy',
    'Name of the policy to use for inference',

)
_USER_ID = flags.DEFINE_integer('user_id', None, 'User ID')
_USER = flags.DEFINE_string('user', None, 'User')
_USE_XVFB = flags.DEFINE_bool('use_xvfb', False, 'Use xvfb')
_DIFFICULTY_LEVELS = flags.DEFINE_list(
    'difficulty_levels', None, 'Difficulty levels to run'
)
_MAX_VOCAB_SIZE = flags.DEFINE_integer(
    'max_vocab_size', 500, 'Max vocab size for web navigation.'
)
_LATENT_DIM = flags.DEFINE_integer(
    'latent_dim', 50, 'Latent dimension for web navigation.'
)
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', 100, 'Embedding dimension for web navigation.'
)
_PROFILE_VALUE_DROPOUT = flags.DEFINE_float(
    'profile_value_dropout', 1.0, 'Profile value dropout for web navigation.'
)
_NUM_WEBSITES = flags.DEFINE_list('num_websites', None, 'Number of websites')
_MOTION_FILES = flags.DEFINE_list('motion_files', None, 'Motion files to run')
_NETLISTS = flags.DEFINE_list('netlists', None, 'Netlists to run')
_STD_CELL_PLACER_MODE = flags.DEFINE_enum(
    'std_cell_placer_mode',
    'dreamplace',
    ['dreamplace', 'fd'],
    'Mode for std cell placer',
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', 'quadruped_locomotion', 'Name of experiment'
)
_EXPERIMENT_ID = flags.DEFINE_string('experiment_id', None, 'Experiment number')
_EXPERIMENT_IDS = flags.DEFINE_list(
    'experiment_ids', None, 'Experiment ids to use for inference'
)
_INTERACTIVE = flags.DEFINE_bool(
    'interactive', False, 'Whether to run in interactive mode'
)
_LOCAL = flags.DEFINE_bool('local', False, 'Run locally or on cluster')
_MODE = flags.DEFINE_enum(
    'mode', 'train', ['train', 'inference', 'generalization'],
    'Mode of execution'
)
_JOB_TYPE = flags.DEFINE_enum(
    'job_type', None,
    ['train', 'collect', 'inference', 'reverb', 'generalization'], 'Type of job'
)
_RUN_OFFLINE_METRICS_ONLY = flags.DEFINE_bool(
    'run_offline_metrics_only', False, 'Whether to run train or inference.'
)
_SEEDS = flags.DEFINE_list('seeds', None, 'Seeds to run')
_SKILL_LEVELS = flags.DEFINE_list(
    'skill_levels', ['novice'], 'Skill levels to run'
)
_TASKS = flags.DEFINE_list('tasks', None, 'Tasks to run')
_TRAIN_EXP_ID = flags.DEFINE_string(
    'train_exp_id',
    None,
    'Experiment where the training logs are stored. This must be present for'
    ' inference or running offline metrics',
)

_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    '127.0.0.1',
    'Address of the variable container server',
)
_VARIABLE_CONTAINER_SERVER_PORT = flags.DEFINE_integer(
    'variable_container_server_port',
    '8008',
    'Port of the variable container server',
)
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address',
    '127.0.0.1',
    'Address of the replay buffer server',
)
_REPLAY_BUFFER_SERVER_PORT = flags.DEFINE_integer(
    'replay_buffer_server_port', '8008', 'Port of the replay buffer server'
)
_VOCABULARY_SERVER_ADDRESS = flags.DEFINE_string(
    'vocabulary_server_address', '127.0.0.1', 'Address of the vocabulary server'
)
_VOCABULARY_SERVER_PORT = flags.DEFINE_integer(
    'vocabulary_server_port', '50000', 'Port of the vocabulary server'
)


def _get_docker_instructions(uid, user, env_name):
  repo_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

  common_setup = [
      """
      ARG APT_COMMAND="apt-get -o Acquire::Retries=3 \
        --no-install-recommends -y"
      """,
      'ENV DEBIAN_FRONTEND=noninteractive',
      'ENV TZ=America/New_York',
      # Install basic system dependencies
      """
      RUN ${APT_COMMAND} update --allow-releaseinfo-change && \
        ${APT_COMMAND} install sudo \
        wget \
        software-properties-common \
        curl \
        tmux \
        telnet \
        net-tools \
        vim \
        less \
        unzip && \
        rm -rf /var/lib/apt/lists/*
        """,
      """
      RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
        ${APT_COMMAND} install -y g++-11
      """,
      # Set up user with the specified ID
      f"""
       RUN if ! getent passwd {uid}; then \
             useradd -m -u {uid} {user}; \
           else \
             existing_user=$(getent passwd {uid} | cut -d: -f1); \
             if [ "{user}" != "$existing_user" ]; then \
               usermod -l {user} $existing_user; \
               usermod -d /home/{user} -m {user}; \
             fi; \
           fi
       """,
      f"""
       RUN echo "{user} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
       """,
  ]

  docker_instructions = {
      'quadruped_locomotion': common_setup + [
          # Set up python3.9 and install requirements for A2perf
          'RUN mkdir -p /workdir',
          'WORKDIR /workdir',
          f'COPY {repo_dir}/quadruped_locomotion_environment.yml .',
          """
          RUN conda update -n base -c conda-forge conda -y && \
            conda env create -f /workdir/quadruped_locomotion_environment.yml --name py39 -y
          """,
          f'COPY {repo_dir} .',
          f"""
            RUN chown -R {uid}:root /workdir && \
             /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                conda activate py39 && \
                pip install -e /workdir[all] seaborn matplotlib minari==0.4.3 && \
                python /workdir/setup.py install && \
                pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11"       
          """,
      ],
      'web_navigation': common_setup + [
          # Install Google Chrome
          'ARG CHROME_VERSION="120.0.6099.109-1"',
          'ARG CHROMEDRIVER_VERSION="120.0.6099.109"',
          """
          RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb && \
            ${APT_COMMAND} update && \
            ${APT_COMMAND} --fix-broken install && \
            ${APT_COMMAND} install /tmp/chrome.deb xvfb && \
            rm /tmp/chrome.deb && \
            rm -rf /var/lib/apt/lists/*
          """,
          # Set up python3.10 and install requirements for A2perf
          'RUN mkdir -p /workdir',
          'WORKDIR /workdir',
          f'COPY {repo_dir}/web_navigation_environment.yml .',
          """
          RUN conda update -n base -c conda-forge conda -y && \
            conda env create -f /workdir/web_navigation_environment.yml --name py310 -y
          """,
          f'COPY {repo_dir} .',
          f"""
            RUN chown -R {uid}:root /workdir && \
             /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                conda activate py310 && \
                pip install -e /workdir[all] seaborn matplotlib chromedriver-py==$CHROMEDRIVER_VERSION minari==0.4.3 && \
                python /workdir/setup.py install && \
                pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11"
            """,
          f'RUN mkdir -p /var/run/dbus && chown -R {uid}:root /var/run/dbus',
          'ENV CONDA_DEFAULT_ENV=py310',
      ],
      'circuit_training': common_setup + [
          # Install dreamplace dependencies
          """
          RUN ${APT_COMMAND} update --allow-releaseinfo-change && \
            ${APT_COMMAND} install flex \
            libcairo2-dev \
            libboost-all-dev && \
            rm -rf /var/lib/apt/lists/*
          """,
          # Set up python3.10 and install requirements for A2perf
          'RUN mkdir -p /workdir',
          'WORKDIR /workdir',
          f'COPY {repo_dir}/circuit_training_environment.yml .',
          """
          RUN conda update -n base -c conda-forge conda -y && \
            conda env create -f /workdir/circuit_training_environment.yml --name py310 -y
          """,
          f'COPY {repo_dir} .',
          """
          RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
              conda activate py310 && \
              pip install -e /workdir[all] seaborn matplotlib minari==0.4.3 && \
              python /workdir/setup.py install && \
              pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11"
          """,
          'ENV CONDA_DEFAULT_ENV=py310',
      ],
  }

  return docker_instructions[env_name]


def _get_entrypoint(domain):
  entrypoints = {
      'quadruped_locomotion': xm.CommandList([
          'echo $@',
          f"""
          /bin/bash <<EOF
          source /opt/conda/etc/profile.d/conda.sh &&
          conda activate py39 &&
          python /workdir/launch/entrypoint.py $@ --verbosity={logging.get_verbosity()}
          EOF
                  """,
          # Waste the trailing "$@" argument
          'echo',
      ]),
      'web_navigation': xm.CommandList([
          'echo $@',
          'service dbus start',
          f"""
                            su {_USER.value} -c /bin/bash <<EOF
          source /opt/conda/etc/profile.d/conda.sh &&
          conda activate py310 &&
          python /workdir/launch/entrypoint.py $@ --verbosity={logging.get_verbosity()}
          EOF
                    """,
          'echo',
      ]),
      'circuit_training': xm.CommandList([
          'echo $@',
          # f"""
          # /bin/bash <<EOF
          # source /opt/conda/etc/profile.d/conda.sh &&
          # conda activate py310 &&
          # python /workdir/launch/entrypoint.py $@ --verbosity={logging.get_verbosity()}
          # EOF
          # """,
          'echo',
      ]),
  }
  return entrypoints[domain]


ENV_NAMES = {
    'quadruped_locomotion': 'QuadrupedLocomotion-v0',
    'web_navigation': 'WebNavigation-v0',
    'circuit_training': 'CircuitTraining-v0',
}

BASE_IMAGE = {
    'quadruped_locomotion': (
        'gcr.io/deeplearning-platform-release/base-gpu:latest'
    ),
    'web_navigation': 'gcr.io/deeplearning-platform-release/base-gpu:latest',
    'circuit_training': 'gcr.io/deeplearning-platform-release/base-gpu:latest',
}

ENV_VARS = {
    'quadruped_locomotion': {
        'PYTHONBUFFERED': '1',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'WRAPT_DISABLE_EXTENSIONS': 'true',
    },
    'web_navigation': {
        'PYTHONBUFFERED': '1',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'WRAPT_DISABLE_EXTENSIONS': 'true',
    },
    'circuit_training': {
        'PYTHONBUFFERED': '1',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'WRAPT_DISABLE_EXTENSIONS': 'true',
    },
}

TASK_TO_MAX_SEQUENCE_LENGTH = dict(
    circuit_training=dict(
        netlist_toy_macro_stdcell_std_cell_placer_mode_dreamplace=3,
        netlist_toy_macro_stdcell_std_cell_placer_mode_fd=3,
        netlist_ariane_std_cell_placer_mode_dreamplace=134,
        netlist_ariane_std_cell_placer_mode_fd=134,
    ),
    quadruped_locomotion=dict(
        dog_pace=100,
        dog_trot=100,
        dog_spin=100,
    ),
    web_navigation=dict(
        difficulty_level_1_num_websites_1=25,
        difficulty_level_1_num_websites_5=25,
        difficulty_level_1_num_websites_10=25,
        difficulty_level_1_num_websites_100=25,
    ),
)


def create_experiment_name(hparams):
  """Creates an experiment name from a dictionary of hyperparameters."""
  return '_'.join(
      f'{key}_{hparams[key]}'
      for key in sorted(hparams.keys())
      if key in ['seed', 'domain', 'algo', 'task', 'skill_level']
  )


def get_web_navigation_hparam_sweeps(**kwargs):
  algos = kwargs['algos']
  mode = kwargs['mode']
  difficulty_levels = kwargs['difficulty_levels']
  num_websites = kwargs['num_websites']
  task_names = [
      f'difficulty_level_{difficulty_level}_num_websites_{sites}'
      for difficulty_level in difficulty_levels
      for sites in num_websites
  ]
  general_hyperparameters = {
      'env_name': ['WebNavigation-v0'],
  }

  latent_dim = kwargs['latent_dim']
  embedding_dim = kwargs['embedding_dim']
  profile_value_dropout = kwargs['profile_value_dropout']
  max_vocab_size = kwargs['max_vocab_size']
  use_xvfb = kwargs['use_xvfb']

  if mode == 'inference':
    general_hyperparameters['gin_config'] = [
        os.path.join(
            '/workdir/a2perf/submission/configs/web_navigation',
            task_name,
            'inference.gin',
        )
        for task_name in task_names
    ]
  elif mode == 'train':
    general_hyperparameters['gin_config'] = [
        '/workdir/a2perf/submission/configs/web_navigation/train.gin'
    ]
  elif mode == 'generalization':
    general_hyperparameters['gin_config'] = [
        '/workdir/a2perf/submission/configs/web_navigation/generalization.gin'
    ]
  elif mode in ['generate', 'evaluate']:
    # gin file does not matter for these modes since we are not running
    # a2perf/submission/main_submission.py
    general_hyperparameters['gin_config'] = ['']
  else:
    raise ValueError(f'Unknown mode: {mode}')

  general_keys, general_values = zip(*general_hyperparameters.items())
  general_hyperparameters = [
      dict(zip(general_keys, v)) for v in
      itertools.product(*general_values)
  ]

  task_hyperparameters = {
      'difficulty_level_1_num_websites_1': {
          'task_name': ['difficulty_level_1_num_websites_1'],
          'difficulty_level': [1, ],
          'num_websites': [1],
      },
      'difficulty_level_1_num_websites_5': {
          'task_name': ['difficulty_level_1_num_websites_5'],
          'difficulty_level': [1],
          'num_websites': [5],
      },
      'difficulty_level_1_num_websites_10': {
          'task_name': ['difficulty_level_1_num_websites_10'],
          'difficulty_level': [1],
          'num_websites': [10],
      },
      'difficulty_level_1_num_websites_100': {
          'task_name': ['difficulty_level_1_num_websites_100'],
          'difficulty_level': [1],
          'num_websites': [100],
      },
  }

  task_sweeps = []
  for task_name, hparams in task_hyperparameters.items():
    if task_name in task_names:
      for params in itertools.product(*hparams.values()):
        task_sweeps.append(
            dict(zip(hparams.keys(), params))
        )
  task_hyperparameters = task_sweeps

  algo_hyperparameters = {
      'bc': {
          'algo': ['bc'],
          'batch_size': [128],
          'learning_rate': [1e-4],
          'env_batch_size': [1],  # still need this to init locomotion env
          'num_epochs': [0],
          'num_iterations': [5000],
          'train_checkpoint_interval': [500],
          'policy_checkpoint_interval': [500],
          'log_interval': [50],
          'eval_interval': [50],
          'max_vocab_size': [max_vocab_size],
          'latent_dim': [latent_dim],
          'embedding_dim': [embedding_dim],
          'profile_value_dropout': [profile_value_dropout],
      },
      'ppo': {
          'algo': ['ppo'],
          'batch_size': [128],
          'entropy_regularization': [1e-2],
          'env_batch_size': [512],
          'eval_interval': [8],
          'learning_rate': [1e-5],
          'log_interval': [8],
          'num_episodes_per_iteration': [512],
          'num_epochs': [4],
          'num_iterations': [8000],
          'policy_checkpoint_interval': [80],
          'train_checkpoint_interval': [80],
          'max_vocab_size': [max_vocab_size],
          'latent_dim': [latent_dim],
          'embedding_dim': [embedding_dim],
          'profile_value_dropout': [profile_value_dropout],
          'use_gae': [False],
      },
      'ddqn': {
          'algo': ['ddqn'],
          'batch_size': [256],
          'learning_rate': [4e-5],
          'epsilon_greedy': [0.3],
          'num_iterations': [100000],
          'rb_capacity': [10000000],
          'env_batch_size': [512],
          'eval_interval': [1000],
          'log_interval': [1000],
          'policy_checkpoint_interval': [10000],
          'train_checkpoint_interval': [10000],
          'max_vocab_size': [max_vocab_size],
          'latent_dim': [latent_dim],
          'embedding_dim': [embedding_dim],
          'profile_value_dropout': [profile_value_dropout],
      },
  }

  algo_sweeps = []
  for algo, hparams in algo_hyperparameters.items():
    if algo in algos:
      for params in itertools.product(*hparams.values()):
        algo_sweeps.append(
            dict(zip(hparams.keys(), params))
        )
  algo_hyperparameters = algo_sweeps

  hyperparameters = []
  for params in itertools.product(
      general_hyperparameters, task_hyperparameters, algo_hyperparameters
  ):
    combined_dict = {
        **params[0],
        **params[1],
        **params[2],
    }
    hyperparameters.append(combined_dict)
  return hyperparameters


def get_quadruped_locomotion_hparam_sweeps(**kwargs):
  algos = kwargs['algos']
  mode = kwargs['mode']
  motion_files = kwargs['motion_files']
  task_names = [motion_file for motion_file in motion_files]
  general_hyperparameters = {
      'env_name': ['QuadrupedLocomotion-v0'],
      # 'motion_file': motion_files,
  }
  if mode == 'inference':
    general_hyperparameters['gin_config'] = [
        os.path.join(
            '/workdir/a2perf/submission/configs/quadruped_locomotion',
            motion_file,
            'inference.gin',
        )
        for motion_file in motion_files
    ]
  elif mode == 'generalization':
    general_hyperparameters['gin_config'] = [
        '/workdir/a2perf/submission/configs/quadruped_locomotion/generalization.gin'
    ]
  elif mode == 'train':
    general_hyperparameters['gin_config'] = [
        '/workdir/a2perf/submission/configs/quadruped_locomotion/train.gin'
    ]
  elif mode in ['generate', 'evaluate']:
    # gin file does not matter for these modes since we are not running
    # a2perf/submission/main_submission.py
    general_hyperparameters['gin_config'] = ['']
  else:
    raise ValueError(f'Unknown mode: {mode}')

  general_keys, general_values = zip(*general_hyperparameters.items())
  general_hyperparameters = [
      dict(zip(general_keys, v)) for v in
      itertools.product(*general_values)
  ]

  task_hyperparameters = {
      'dog_pace': {
          'task_name': ['dog_pace'],
          'motion_file_path': [
              '/workdir/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt'
          ],
      },
      'dog_trot': {
          'task_name': ['dog_trot'],
          'motion_file_path': [
              '/workdir/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_trot.txt'
          ],
      },
      'dog_spin': {
          'task_name': ['dog_spin'],
          'motion_file_path': [
              '/workdir/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_spin.txt'
          ],
      },
  }

  task_sweeps = []
  for task_name, hparams in task_hyperparameters.items():
    if task_name in task_names:
      for params in itertools.product(*hparams.values()):
        task_sweeps.append(
            dict(zip(hparams.keys(), params))
        )
  task_hyperparameters = task_sweeps

  algo_hyperparameters = {
      'bc':
        {
            'algo': ['bc'],
            'batch_size': [64],
            'learning_rate': [1e-4],
            'env_batch_size': [0],
            'num_epochs': [0],
            'num_iterations': [1000],
            'train_checkpoint_interval': [10],
            'policy_checkpoint_interval': [10],
            'log_interval': [1],
            'eval_interval': [1],
        },
      'ppo': {
          'algo': ['ppo'],
          'batch_size': [128],
          'entropy_regularization': [1e-2],
          'env_batch_size': [512],
          'eval_interval': [8],
          'learning_rate': [1e-5],
          'log_interval': [8],
          'num_episodes_per_iteration': [512],
          'num_epochs': [4],
          'num_iterations': [8000],
          'policy_checkpoint_interval': [80],
          'train_checkpoint_interval': [80],
          'use_gae': [False],
      },
      'sac': {
          'algo': ['sac'],
          'batch_size': [256],
          'learning_rate': [3e-4],
          'num_iterations': [2000000],
          'rb_capacity': [2000000],
          'env_batch_size': [512],
          'eval_interval': [2000],
          'log_interval': [2000],
          'policy_checkpoint_interval': [20000],
          'train_checkpoint_interval': [20000],
      },
      'td3': {
          'algo': ['td3'],
          'batch_size': [256],
          'learning_rate': [3e-4],
          'num_iterations': [50000],
          'rb_capacity': [1000000],
          'env_batch_size': [512],
          'eval_interval': [50],
          'log_interval': [50],
          'policy_checkpoint_interval': [500],
          'train_checkpoint_interval': [500],
      },
      'ddpg': {
          'algo': ['ddpg'],
          'batch_size': [256],
          'learning_rate': [4e-4],
          'num_iterations': [50000],
          'rb_capacity': [1000000],
          'env_batch_size': [512],
          'eval_interval': [50],
          'log_interval': [50],
          'policy_checkpoint_interval': [500],
          'train_checkpoint_interval': [500],
      },
  }

  algo_sweeps = []
  for algo, hparams in algo_hyperparameters.items():
    if algo in algos:
      for params in itertools.product(*hparams.values()):
        algo_sweeps.append(
            dict(zip(hparams.keys(), params))
        )
  algo_hyperparameters = algo_sweeps

  print(f'The algo_hyperparameters are: {algo_hyperparameters}')
  print(f'the task_hyperparameters are: {task_hyperparameters}')
  hyperparameters = []
  for params in itertools.product(
      general_hyperparameters, task_hyperparameters, algo_hyperparameters
  ):
    combined_dict = {
        **params[0],
        **params[1],
        **params[2],
    }
    hyperparameters.append(combined_dict)
  return hyperparameters


def get_circuit_training_hparam_sweeps(**kwargs):
  algos = kwargs['algos']
  mode = kwargs['mode']
  netlists = kwargs['netlists']
  std_cell_placer_mode = _STD_CELL_PLACER_MODE.value
  task_names = [
      f'netlist_{netlist}_std_cell_placer_mode_{std_cell_placer_mode}'
      for netlist in netlists
  ]

  general_hyperparameters = {
      'env_name': ['CircuitTraining-v0'],
  }

  if mode == 'inference':
    general_hyperparameters['gin_config'] = [
        os.path.join(
            '/workdir/a2perf/submission/configs/circuit_training',
            netlist,
            'inference.gin',
        )
        for netlist in netlists
    ]
  elif mode == 'generalization':
    general_hyperparameters['gin_config'] = [
        '/workdir/a2perf/submission/configs/circuit_training/generalization.gin'
    ]
  elif mode == 'train':
    general_hyperparameters['gin_config'] = [
        '/workdir/a2perf/submission/configs/circuit_training/train.gin'
    ]

  elif mode in ['generate', 'evaluate']:
    # gin file doesn't really matter for these modes since we are not running
    # a2perf/submission/main_submission.py
    general_hyperparameters['gin_config'] = ['']
  else:
    raise ValueError(f'Unknown mode: {mode}')

  general_keys, general_values = zip(*general_hyperparameters.items())
  general_hyperparameters = [
      dict(zip(general_keys, v)) for v in
      itertools.product(*general_values)
  ]

  task_hyperparameters = {
      'netlist_toy_macro_stdcell_std_cell_placer_mode_dreamplace': {
          'task_name': [
              'netlist_toy_macro_stdcell_std_cell_placer_mode_dreamplace'
          ],
          'netlist_path': [
              os.path.join(
                  '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data',
                  'toy_macro_stdcell',
                  'netlist.pb.txt',
              ),
          ],
          'std_cell_placer_mode': ['dreamplace'],
          'init_placement_path': [
              '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/initial.plc'
          ],
      },
      'netlist_toy_macro_stdcell_std_cell_placer_mode_fd': {
          'task_name': ['netlist_toy_macro_stdcell_std_cell_placer_mode_fd'],
          'netlist_path': [
              os.path.join(
                  '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data',
                  'toy_macro_stdcell',
                  'netlist.pb.txt',
              )
          ],
          'std_cell_placer_mode': ['fd'],
          'init_placement_path': [
              '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/initial.plc'
          ],
      },
      'netlist_ariane_std_cell_placer_mode_dreamplace': {
          'task_name': ['netlist_ariane_std_cell_placer_mode_dreamplace'],
          'netlist_path': [
              os.path.join(
                  '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data',
                  'ariane',
                  'netlist.pb.txt',
              )
          ],
          'std_cell_placer_mode': ['dreamplace'],
          'init_placement_path': [
              '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data/ariane/initial.plc'
          ],
      },
      'netlist_ariane_std_cell_placer_mode_fd': {
          'task_name': ['netlist_ariane_std_cell_placer_mode_fd'],
          'netlist_path': [
              os.path.join(
                  '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data',
                  'ariane',
                  'netlist.pb.txt',
              )
          ],
          'std_cell_placer_mode': ['dreamplace'],
          'init_placement_path': [
              '/workdir/a2perf/domains/circuit_training/circuit_training/environment/test_data/ariane/initial.plc'
          ],
      },
  }

  task_sweeps = []
  for task_name, hparams in task_hyperparameters.items():
    if task_name in task_names:
      for params in itertools.product(*hparams.values()):
        task_sweeps.append(
            dict(zip(hparams.keys(), params))
        )
  task_hyperparameters = task_sweeps

  # Different tasks may have different hparams for the same algo
  algo_hyperparameters = {
      'netlist_toy_macro_stdcell_std_cell_placer_mode_dreamplace': {
          'bc': {
              'algo': ['bc'],
              'batch_size': [128],
              'learning_rate': [1e-3],
              'env_batch_size': [1],  # still need this to init locomotion env
              'num_epochs': [0],
              'num_iterations': [1000],
              # 'num_iterations': [10],
              'train_checkpoint_interval': [100],
              'policy_checkpoint_interval': [100],
              'log_interval': [10],
              'eval_interval': [10],
          },
          'ppo': {
              'algo': ['ppo'],
              'batch_size': [128],
              'entropy_regularization': [1e-2],
              'env_batch_size': [512],
              'eval_interval': [5],
              'learning_rate': [4e-4],
              'log_interval': [5],
              'num_episodes_per_iteration': [32],
              'num_epochs': [6],
              'num_iterations': [5000],
              'policy_checkpoint_interval': [50],
              'train_checkpoint_interval': [50],
              'use_gae': [False],
          },
          'ddqn': {
              'algo': ['ddqn'],
              'batch_size': [256],
              'env_batch_size': [512],
              'epsilon_greedy': [0.3],
              'eval_interval': [10],
              'learning_rate': [4e-5],
              'log_interval': [10],
              'num_iterations': [10000],
              'policy_checkpoint_interval': [100],
              'rb_capacity': [1000000],
              'train_checkpoint_interval': [100],
          },
      },
      'netlist_ariane_std_cell_placer_mode_dreamplace': {
          'bc': {
              'algo': ['bc'],
              'batch_size': [128],
              'learning_rate': [1e-3],
              'env_batch_size': [1],  # still need this to init locomotion env
              'num_epochs': [0],
              'num_iterations': [5000],
              'train_checkpoint_interval': [500],
              'policy_checkpoint_interval': [500],
              'log_interval': [50],
              'eval_interval': [50],
          },
          'ppo': {
              'algo': ['ppo'],
              # 'batch_size': [128],
              'batch_size': [32],
              'entropy_regularization': [1e-2],
              # 'env_batch_size': [512],
              'env_batch_size': [4],
              'eval_interval': [1],
              'learning_rate': [4e-4],
              'log_interval': [1],
              # 'num_episodes_per_iteration': [1024],
              'num_episodes_per_iteration': [8],
              'num_epochs': [4],
              'num_iterations': [250],
              'policy_checkpoint_interval': [25],
              'train_checkpoint_interval': [25],
              'use_gae': [False],
          },
          'ddqn': {
              'algo': ['ddqn'],
              'batch_size': [256],
              'learning_rate': [4e-5],
              'epsilon_greedy': [0.3],
              'num_iterations': [100000],
              'rb_capacity': [10000000],
              'env_batch_size': [512],
              'eval_interval': [1000],
              'log_interval': [1000],
              'policy_checkpoint_interval': [10000],
              'train_checkpoint_interval': [10000],
          },
      },
  }

  algo_sweeps = []
  for task, algo_hparams in algo_hyperparameters.items():
    if task in task_names:
      for algo, hparams in algo_hparams.items():
        if algo in algos:
          for params in itertools.product(*hparams.values()):
            algo_sweeps.append(
                dict(zip(hparams.keys(), params))
            )
  algo_hyperparameters = algo_sweeps

  hyperparameters = []
  for params in itertools.product(
      general_hyperparameters, task_hyperparameters, algo_hyperparameters
  ):
    combined_dict = {
        **params[0],
        **params[1],
        **params[2],
    }
    hyperparameters.append(combined_dict)
  return hyperparameters


def get_hparam_sweeps(domain, **kwargs):
  skill_levels = kwargs['skill_levels']
  seeds = kwargs['seeds']
  debug = kwargs['debug']
  mode = kwargs['mode']

  if domain == 'quadruped_locomotion':
    hparam_sweeps = get_quadruped_locomotion_hparam_sweeps(**kwargs)
  elif domain == 'web_navigation':
    hparam_sweeps = get_web_navigation_hparam_sweeps(**kwargs)
  elif domain == 'circuit_training':
    hparam_sweeps = get_circuit_training_hparam_sweeps(**kwargs)
  else:
    raise ValueError(f'Unknown domain: {domain}')

  hyperparameters = dict(
      domain=[domain],
      debug=[debug],
      skill_level=skill_levels,
      seed=seeds,
      mode=[mode],
  )
  hyperparameter_sweeps = []
  for param in itertools.product(*hyperparameters.values()):
    hyperparameter_sweeps.append(dict(zip(hyperparameters.keys(), param)))

  # Now combine the hyperparameters with the hparam_sweeps for final sweeps
  final_sweeps = []
  for params in itertools.product(hyperparameter_sweeps, hparam_sweeps):
    combined_dict = {
        **params[0],
        **params[1],
    }
    final_sweeps.append(combined_dict)
  return final_sweeps


def main(_):
  create_experiment = xm_local.create_experiment

  with create_experiment(experiment_title=_EXPERIMENT_NAME.value) as experiment:

    experiment_id = experiment.experiment_id
    experiment_id = _EXPERIMENT_ID.value or experiment_id

    base_root_dir = os.path.join(
        '/gcs',
        'a2perf',
        'experiments',
        _DOMAIN.value,
        str(experiment_id),
    )

    async def make_job(work_unit: xm.WorkUnit, **hparams):
      task = hparams['task_name']
      hparams['num_replicas'] = _NUM_GPUS.value
      hparams['num_collect_jobs_per_machine'] = 1
      hparams['max_sequence_length'] = \
        TASK_TO_MAX_SEQUENCE_LENGTH[_DOMAIN.value][task]
      executor = xm_local.Local(
          requirements=xm.JobRequirements(
              resources={
                  xm.ResourceType.LOCAL_GPU: _NUM_GPUS.value,
              },
          ),
          docker_options=xm_local.DockerOptions(
              ports={
                  _REPLAY_BUFFER_SERVER_PORT.value: (
                      _REPLAY_BUFFER_SERVER_PORT.value
                  ),
                  _VARIABLE_CONTAINER_SERVER_PORT.value: (
                      _VARIABLE_CONTAINER_SERVER_PORT.value
                  ),
                  _VOCABULARY_SERVER_PORT.value: _VOCABULARY_SERVER_PORT.value,
              },
              volumes=None,
              mount_gcs_path=True,
              interactive=_INTERACTIVE.value,
          ),
          experimental_stream_output=True,
      )
      docker_instructions = _get_docker_instructions(
          uid=_USER_ID.value, env_name=_DOMAIN.value, user=_USER.value
      )

      base_image = BASE_IMAGE[_DOMAIN.value]
      if _NUM_GPUS.value == 0:
        base_image = 'gcr.io/deeplearning-platform-release/base-cpu:latest'

      # Define Executable
      [executable] = experiment.package([
          xm.python_container(
              executor_spec=executor.Spec(),
              path='.',
              use_deep_module=True,
              base_image=base_image,
              docker_instructions=docker_instructions,
              entrypoint=_get_entrypoint(_DOMAIN.value),
              env_vars=ENV_VARS[_DOMAIN.value],
          )
      ])

      hparams['root_dir'] = os.path.join(
          hparams['root_dir'],
          str(work_unit.work_unit_id),
      )

      job = xm.Job(executable, args=hparams, executor=executor)
      work_unit.add(job)

      # Also save the hparam config to the experiment directory
      hparam_config_path = os.path.join(
          '~/gcs',
          'a2perf',
          _DOMAIN.value,
          str(experiment_id),
          'hparam_config.json',
      )

      # Turn it into absolute path
      hparam_config_path = os.path.expanduser(hparam_config_path)

      # Make the dir and write the hparam config
      os.makedirs(os.path.dirname(hparam_config_path), exist_ok=True)
      with open(hparam_config_path, 'w') as f:
        f.write(str(hparams))

    hparam_sweeps = get_hparam_sweeps(
        debug=_DEBUG.value,
        tasks=_TASKS.value,
        algos=_ALGOS.value,
        motion_files=_MOTION_FILES.value,
        num_websites=_NUM_WEBSITES.value,
        difficulty_levels=_DIFFICULTY_LEVELS.value,
        netlists=_NETLISTS.value,
        domain=_DOMAIN.value,
        skill_levels=_SKILL_LEVELS.value,
        seeds=_SEEDS.value,
        use_xvfb=_USE_XVFB.value,
        mode=_MODE.value,
        latent_dim=_LATENT_DIM.value,
        embedding_dim=_EMBEDDING_DIM.value,
        profile_value_dropout=_PROFILE_VALUE_DROPOUT.value,
        max_vocab_size=_MAX_VOCAB_SIZE.value,
    )
    print(f'The hparam sweeps are: {hparam_sweeps}')
    for hparams in hparam_sweeps:
      experiment_name = create_experiment_name(hparams)
      task_name = hparams['task_name']
      hparams['root_dir'] = os.path.join(
          base_root_dir, task_name, hparams['algo'], experiment_name
      )

      skill_level = hparams['skill_level']
      domain = hparams['domain']
      mode = hparams['mode']
      env_name = ENV_NAMES[domain][:-3]
      dataset_id = f'{env_name}-{task_name}-{skill_level}-v0'

      print(dataset_id)
      hparams.update(
          dict(
              policy_name=_POLICY_NAME.value,
              vocabulary_manager_auth_key=_VOCABULARY_MANAGER_AUTH_KEY.value,
              job_type=_JOB_TYPE.value,
              experiment_id=experiment_id,
              replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
              replay_buffer_server_port=_REPLAY_BUFFER_SERVER_PORT.value,
              variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
              variable_container_server_port=_VARIABLE_CONTAINER_SERVER_PORT.value,
              vocabulary_server_address=_VOCABULARY_SERVER_ADDRESS.value,
              vocabulary_server_port=_VOCABULARY_SERVER_PORT.value,
              run_offline_metrics_only=_RUN_OFFLINE_METRICS_ONLY.value,
              dataset_id=dataset_id,
              datasets_path=_DATASETS_PATH.value,
              participant_module_path=os.path.join(
                  '/workdir/a2perf/a2perf_benchmark_submission',
              ),
          )
      )

      experiment.add(make_job, args=hparams)


if __name__ == '__main__':
  app.run(main)
