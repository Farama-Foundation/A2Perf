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
from xmanager import xm
from xmanager import xm_local

_ALGOS = flags.DEFINE_list(
    'algos',
    ['ppo'],
    'Algorithms to run. If multiple are specified, they will be run in'
    ' sequence',
)
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to use')

_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
_DOMAIN = flags.DEFINE_enum(
    'domain',
    'quadruped_locomotion',
    ['quadruped_locomotion', 'web_navigation', 'circuit_training'],
    'Domain to run',
)
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

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', 'quadruped_locomotion', 'Name of experiment'
)
_EXPERIMENT_NUMBER = flags.DEFINE_string(
    'experiment_number', None, 'Experiment number'
)
_INFERENCE = flags.DEFINE_bool(
    'inference', False, 'Whether to run train or inference.'
)
_INTERACTIVE = flags.DEFINE_bool(
    'interactive', False, 'Whether to run in interactive mode'
)
_LOCAL = flags.DEFINE_bool('local', False, 'Run locally or on cluster')
_MODE = flags.DEFINE_enum(
    'mode', 'train', ['train', 'inference'], 'Mode of execution'
)
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    'participant_module_path', None, 'Path to participant module'
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

REPO_DIR = os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DOCKER_INSTRUCTIONS = {
    'quadruped_locomotion': [
        '''ARG APT_COMMAND="sudo apt-get -o Acquire::Retries=3 \
          --no-install-recommends -y"''',
        'ENV DEBIAN_FRONTEND=noninteractive',
        'RUN ${APT_COMMAND} update && ${APT_COMMAND} install sudo wget unzip',
        # Delete user with UID 1000 and then create a new user 'user' with UID 1000
        f"""
        RUN userdel $(getent passwd {os.getuid()} | cut -d: -f1) || true \
          && useradd -m -u {os.getuid()} user""",
        'RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers',
        'USER user',
        'RUN sudo mkdir -p /workdir',
        'WORKDIR /workdir',
        # Set up custom conda environment
        'RUN conda create -y --name py39 python=3.9',
        'ENV CONDA_DEFAULT_ENV=py39',
        'ENV PATH="/opt/conda/envs/py39/bin:${PATH}"',
        'RUN /opt/conda/envs/py39/bin/pip install --upgrade pip setuptools',
        """
        RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
          conda activate py39 && \
          conda install cuda -c  nvidia -y"
        """,
        # 'RUN /opt/conda/envs/py39/bin/pip install nvidia-pyindex',
        # 'RUN /opt/conda/envs/py39/bin/pip install nvidia-tensorrt',
        # Install Requirements for A2Perf
        """RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
          conda activate py39 && \
          conda install -c conda-forge -y gcsfs"
          """,
        f"""COPY {REPO_DIR}/a2perf/metrics/reliability/requirements.txt \
          ./a2perf/metrics/reliability/requirements.txt""",
        f"""
        COPY {REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt \
          ./a2perf/metrics/system/codecarbon/
        """,
        f"""
        COPY {REPO_DIR}/a2perf/domains/quadruped_locomotion/requirements.txt \
          ./a2perf/domains/quadruped_locomotion/requirements.txt
        """,
        f"""
        COPY {REPO_DIR}/a2perf/a2perf_benchmark_submission/requirements.txt \
          ./a2perf/a2perf_benchmark_submission/requirements.txt
        """,
        f'COPY {REPO_DIR}/requirements.txt ./requirements.txt',
        'RUN /opt/conda/envs/py39/bin/pip install -r ./requirements.txt',
        (
            'RUN /opt/conda/envs/py39/bin/pip install -r'
            ' ./a2perf/domains/quadruped_locomotion/requirements.txt'
        ),
        (
            'RUN /opt/conda/envs/py39/bin/pip install -r'
            ' ./a2perf/a2perf_benchmark_submission/requirements.txt'
        ),
        f'COPY {REPO_DIR} .',
        'RUN sudo chmod -R 777 /workdir',
        'RUN /opt/conda/envs/py39/bin/pip install /workdir',
    ],
    'web_navigation': [
        '''ARG APT_COMMAND="sudo apt-get -o Acquire::Retries=3 \
          --no-install-recommends -y"''',
        'ENV DEBIAN_FRONTEND=noninteractive',
        'RUN ${APT_COMMAND} update && ${APT_COMMAND} install sudo wget unzip',
        # Set up user with same UID as host user
        f'RUN if ! id {os.getuid()}; then useradd -m -u {os.getuid()} user; fi',
        'RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers',
        'RUN mkdir -p /workdir',
        'WORKDIR /workdir',
        # Set up custom conda environment
        'RUN conda create -y --name py310 python=3.10',
        'ENV CONDA_DEFAULT_ENV=py310',
        'ENV PATH="/opt/conda/envs/py310/bin:${PATH}"',
        'RUN /opt/conda/envs/py310/bin/pip install --upgrade pip setuptools',
        """
        RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
          conda activate py310 && \
          conda install cuda -c  nvidia -y && \
          pip install nvidia-pyindex && \
          pip install nvidia-tensorrt"
        """,
        # Chrome Installation
        'ARG CHROME_VERSION="120.0.6099.109-1"',
        'ARG CHROMEDRIVER_VERSION="120.0.6099.109"',
        """
        RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb && \
          ${APT_COMMAND} update && \
          ${APT_COMMAND} --fix-broken install && \
          ${APT_COMMAND} install /tmp/chrome.deb xvfb && \
          rm /tmp/chrome.deb
        """,
        """
        RUN TODAYS_DATE=$(date +%Y-%m-%d) && \
            wget --no-verbose -O /tmp/chromedriver-linux64.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROMEDRIVER_VERSION}/linux64/chromedriver-linux64.zip && \
            unzip -o /tmp/chromedriver-linux64.zip -d /tmp/ && \
            mv /tmp/chromedriver-linux64/chromedriver /tmp/chromedriver && \
            mkdir -p /home/user/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION} && \
            mv /tmp/chromedriver /home/user/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION}/ && \
            rm /tmp/chromedriver-linux64.zip && \
            printf '{"linux64_chromedriver_%s_for_%s": {"timestamp": "%s", "binary_path": "/home/user/.wdm/drivers/chromedriver/linux64/%s/chromedriver"}}' "${CHROMEDRIVER_VERSION}" "${CHROME_VERSION}" "${TODAYS_DATE}" "${CHROMEDRIVER_VERSION}" > /home/user/.wdm/drivers.json && \
            chmod -R 777 /home/user/.wdm
        """,
        # Install Requirements for A2Perf
        f"""COPY {REPO_DIR}/a2perf/metrics/reliability/requirements.txt \
      ./a2perf/metrics/reliability/requirements.txt""",
        f"""
    COPY {REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt \
      ./a2perf/metrics/system/codecarbon/
    """,
        f"""
    COPY {REPO_DIR}/a2perf/a2perf_benchmark_submission/requirements.txt \
      ./a2perf/a2perf_benchmark_submission/requirements.txt
      """,
        f"""
    COPY {REPO_DIR}/a2perf/domains/web_navigation/requirements.txt \
      ./a2perf/domains/web_navigation/requirements.txt
    """,
        f"""
        COPY {REPO_DIR}/a2perf/domains/web_navigation/gwob/miniwob_plusplus/python/requirements.txt \
          ./a2perf/domains/web_navigation/gwob/miniwob_plusplus/python/requirements.txt
        """,
        f'COPY {REPO_DIR}/requirements.txt ./requirements.txt',
        'RUN /opt/conda/envs/py310/bin/pip install -r ./requirements.txt',
        (
            'RUN /opt/conda/envs/py310/bin/pip install -r'
            ' ./a2perf/domains/web_navigation/requirements.txt'
        ),
        (
            'RUN /opt/conda/envs/py310/bin/pip install -r'
            ' ./a2perf/a2perf_benchmark_submission/requirements.txt'
        ),
        f'COPY {REPO_DIR} .',
        'RUN sudo chmod -R 777 /workdir',
        'RUN /opt/conda/envs/py310/bin/pip install /workdir',
        (
            'ENV PATH="/home/user/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION}:${PATH}"'
        ),
    ],
    'circuit_training': [],
}

ENTRYPOINT = {
    'quadruped_locomotion': xm.CommandList([
        'python /workdir/launch/entrypoint.py',
    ]),
    'web_navigation': xm.CommandList([
        'sudo service dbus start',
        'echo "$@"',
        '/opt/conda/envs/py310/bin/python /workdir/launch/entrypoint.py',
    ]),
    'circuit_training': xm.CommandList([]),
}

ENV_NAMES = {
    'quadruped_locomotion': 'QuadrupedLocomotion-v0',
    'web_navigation': 'WebNavigation-v0',
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
        'TF_GPU_THREAD_MODE': 'gpu_private',
        'TF_USE_LEGACY_KERAS': '1',
        'WRAPT_DISABLE_EXTENSIONS': 'true',
    },
    'web_navigation': {
        'PYTHONBUFFERED': '1',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_GPU_THREAD_MODE': 'gpu_private',
        'TF_USE_LEGACY_KERAS': '1',
        'WRAPT_DISABLE_EXTENSIONS': 'true',
    },
    'circuit_training': {'WRAPT_DISABLE_EXTENSIONS': 'true'},
}


def create_experiment_name(hparams):
  """Creates an experiment name from a dictionary of hyperparameters."""
  return '_'.join(
      f'{key}_{hparams[key]}'
      for key in sorted(hparams.keys())
      if key in ['seed', 'domain', 'algo', 'task', 'skill_level']
  )


def get_hparam_sweeps(domain, **kwargs):
  algos = kwargs['algos']
  skill_levels = kwargs['skill_levels']
  seeds = kwargs['seeds']
  debug = kwargs['debug']
  mode = kwargs['mode']

  if domain == 'quadruped_locomotion':
    motion_files = kwargs['motion_files']
    general_hyperparameters = {
        'batch_size': [64],
        'eval_interval': [100],
        'log_interval': [100],
        'env_name': ['QuadrupedLocomotion-v0'],
        'motion_file': motion_files,
    }

    if debug:
      general_hyperparameters.update({
          'env_batch_size': [8],
          'total_env_steps': [1000000],
          'train_checkpoint_interval': [10000],
          'policy_checkpoint_interval': [10000],
          'timesteps_per_actorbatch': [256],
      })

      algo_hyperparameters = {
          'ppo': {
              'algo': ['ppo'],
              'use_gae': [True],
              'num_epochs': [1],
              'learning_rate': [3e-4],
              'entropy_regularization': [1e-4],
          },
          'sac': {
              'algo': ['sac'],
              'learning_rate': [3e-4],
              'rb_capacity': [100000],
          },
      }
    else:
      general_hyperparameters.update({
          'env_batch_size': [44],
          'total_env_steps': [100000000],
          'train_checkpoint_interval': [1000000],
          'policy_checkpoint_interval': [1000000],
          'timesteps_per_actorbatch': [4096],
      })

      algo_hyperparameters = {
          'ppo': {
              'algo': ['ppo'],
              'use_gae': [True],
              'entropy_regularization': [1e-5],
              'learning_rate': [3e-4],
              'num_epochs': [10],
          },
          'sac': {
              'algo': ['sac'],
              'learning_rate': [3e-4],
              'rb_capacity': [10000000],
          },
      }
  elif domain == 'web_navigation':
    num_websites = kwargs['num_websites']
    difficulty_levels = kwargs['difficulty_levels']
    latent_dim = kwargs['latent_dim']
    embedding_dim = kwargs['embedding_dim']
    profile_value_dropout = kwargs['profile_value_dropout']
    max_vocab_size = kwargs['max_vocab_size']
    use_xvfb = kwargs['use_xvfb']
    general_hyperparameters = {
        'eval_interval': [100],
        'log_interval': [100],
        'env_name': ['WebNavigation-v0'],
        'num_websites': num_websites,
        'difficulty_level': difficulty_levels,
        'max_vocab_size': [max_vocab_size],
        'latent_dim': [latent_dim],
        'embedding_dim': [embedding_dim],
        'profile_value_dropout': [profile_value_dropout],
        'use_xvfb': [use_xvfb],
    }

    if debug:
      general_hyperparameters.update({
          'env_batch_size': [4],
          'total_env_steps': [100000],
          'train_checkpoint_interval': [10000],
          'policy_checkpoint_interval': [10000],
          'timesteps_per_actorbatch': [256],
      })

      algo_hyperparameters = {
          'ppo': {
              'use_gae': [True],
              'algo': ['ppo'],
              'batch_size': [32],
              'num_epochs': [5],
              'learning_rate': [3e-4],
              'entropy_regularization': [1e-4],
          },
          'ddqn': {
              'algo': ['ddqn'],
              'batch_size': [32],
              'epsilon_greedy': [0.1],
              'learning_rate': [3e-4],
              'rb_capacity': [10000],
          },
      }
    else:
      general_hyperparameters.update({
          'env_batch_size': [8],
          'total_env_steps': [10000000],
          'train_checkpoint_interval': [100000],
          'policy_checkpoint_interval': [100000],
          'timesteps_per_actorbatch': [4096],
      })

      algo_hyperparameters = {
          'ppo': {
              'use_gae': [True],
              'algo': ['ppo'],
              'batch_size': [32],
              'entropy_regularization': [1e-4],
              'learning_rate': [3e-4],
              'num_epochs': [10],
          },
          'ddqn': {
              'algo': ['ddqn'],
              'batch_size': [32],
              'learning_rate': [3e-4],
              'epsilon_greedy': [0.1],
              'rb_capacity': [1000000],
          },
      }
  elif domain == 'circuit_training':
    pass

  else:
    raise ValueError(f'Unknown domain: {domain}')

  algo_hparam_combinations = []
  for algo in algos:
    if algo in algo_hyperparameters:
      keys, values = zip(*algo_hyperparameters[algo].items())
      algo_hparam_combinations.extend(
          [dict(zip(keys, v)) for v in itertools.product(*values)]
      )

  general_hyperparameters.update({
      'skill_level': skill_levels,
      'seed': seeds,
      'debug': [debug],
      'mode': [mode],
      'domain': [domain],
  })

  # Generate combinations of general hyperparameters
  general_keys, general_values = zip(*general_hyperparameters.items())
  general_hparam_combinations = [
      dict(zip(general_keys, v)) for v in itertools.product(*general_values)
  ]

  # Combine algorithm-specific hyperparameters with general hyperparameters
  hparam_sweeps = []
  for algo_hparam in algo_hparam_combinations:
    for general_hparam in general_hparam_combinations:
      combined_hparam = {**general_hparam, **algo_hparam}
      hparam_sweeps.append(combined_hparam)

  return hparam_sweeps


def main(_):
  create_experiment = xm_local.create_experiment

  with create_experiment(experiment_title=_EXPERIMENT_NAME.value) as experiment:
    base_root_dir = os.path.join(
        '/gcs',
        'a2perf',
        _DOMAIN.value,
        str(experiment.experiment_id),
    )

    async def make_job(work_unit: xm.WorkUnit, **hparams):
      executor = xm_local.Local(
          requirements=xm.JobRequirements(
              resources={
                  xm.ResourceType.LOCAL_GPU: _NUM_GPUS.value,
              },
          ),
          docker_options=xm_local.DockerOptions(
              ports=None,
              volumes=None,
              mount_gcs_path=True,
              interactive=_INTERACTIVE.value,
          ),
          experimental_stream_output=True,
      )

      # Define Executable
      [executable] = experiment.package([
          xm.python_container(
              executor_spec=executor.Spec(),
              path='../',
              use_deep_module=True,
              base_image=BASE_IMAGE[_DOMAIN.value],
              docker_instructions=DOCKER_INSTRUCTIONS[_DOMAIN.value],
              entrypoint=ENTRYPOINT[_DOMAIN.value],
              env_vars=ENV_VARS[_DOMAIN.value],
          )
      ])

      hparams['root_dir'] = os.path.join(
          hparams['root_dir'],
          str(work_unit.work_unit_id),
      )

      job = xm.Job(executable, args=hparams, executor=executor)
      work_unit.add(job)

    hparam_sweeps = get_hparam_sweeps(
        debug=_DEBUG.value,
        tasks=_TASKS.value,
        algos=_ALGOS.value,
        motion_files=_MOTION_FILES.value,
        num_websites=_NUM_WEBSITES.value,
        difficulty_levels=_DIFFICULTY_LEVELS.value,
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

    for hparams in hparam_sweeps:
      if _DOMAIN.value == 'quadruped_locomotion':
        task = hparams['motion_file']
        hparams.pop('motion_file')
        hparams['motion_file_path'] = os.path.join(
            '/workdir/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/',
            task + '.txt',
        )
      elif _DOMAIN.value == 'web_navigation':
        difficulty_level = hparams['difficulty_level']
        num_websites = hparams['num_websites']
        task = (
            f'difficulty_level_{difficulty_level}_num_websites_{num_websites}'
        )
      else:
        raise ValueError(f'Unknown domain: {_DOMAIN.value}')

      experiment_name = create_experiment_name(hparams)
      experiment_dir = os.path.join(
          base_root_dir,
          task,
          hparams['algo'],
          'debug' if _DEBUG.value else '',
      )

      hparams['root_dir'] = os.path.join(
          experiment_dir,
          experiment_name,
      )

      algo = hparams['algo']
      skill_level = hparams['skill_level']
      domain = hparams['domain']
      debug = hparams['debug']
      mode = hparams['mode']

      dataset_id = f'{domain[0].upper() + domain[1:]}-{task}-{skill_level}-v0'

      hparams.update(
          dict(
              run_offline_metrics_only=_RUN_OFFLINE_METRICS_ONLY.value,
              dataset_id=dataset_id,
              gin_config=os.path.join(
                  '/workdir/a2perf/submission/configs',
                  domain,
                  'debug' if debug else '',
                  f'{mode}.gin',
              ),
              participant_module_path=os.path.join(
                  '/workdir/a2perf/a2perf_benchmark_submission',
                  algo,
              ),
          )
      )

    for hparam_config in hparam_sweeps:
      experiment.add(make_job, args=hparam_config)


if __name__ == '__main__':
  app.run(main)
