"""Launches experiments for the A2Perf benchmark.

Example usage:

  Quadruped Locomotion:
    xmanager launch launch/xm_launch.py -- \
    --domain=quadruped_locomotion \
    --seeds=4 \
    --algos=sac \
    --tasks=dog_pace \
    --mode=train \
    --root_dir=~/gcs # Be sure to make this folder first

  Web Navigation:
    xmanager launch launch/xm_launch.py -- \
    --domain=web_navigation \
    --seeds=4 \
    --algos=sac_lstm \
    --tasks=1 \
    --num_websites=1 \
    --mode=train \
    --root_dir=~/gcs \
    --debug
"""

import itertools
import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', 'quadruped_locomotion', 'Name of experiment'
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir', '/tmp/xm_local', 'Base directory for logs and results'
)
_INTERACTIVE = flags.DEFINE_bool(
    'interactive', False, 'Whether to run in interactive mode'
)
_TRAIN_LOGS_DIRS = flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train'],
    'Directory patterns fr train logs that will be used to calculate reliability metrics. Should be glob patterns',
)
_DIFFICULTY_LEVELS = flags.DEFINE_list(
    'difficulty_levels', None, 'Difficulty levels to run'
)
_DOMAIN = flags.DEFINE_enum(
    'domain',
    'quadruped_locomotion',
    ['quadruped_locomotion', 'web_navigation', 'circuit_training'],
    'Domain to run'
)
_LOCAL = flags.DEFINE_bool('local', False, 'Run locally or on cluster')
_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
_TRAIN_EXP_ID = flags.DEFINE_string(
    'train_exp_id',
    None,
    'Experiment where the training logs are stored. This must be present for'
    ' inference or running offline metrics',
)
_INFERENCE = flags.DEFINE_bool(
    'inference', False, 'Whether to run train or inference.'
)
_RUN_OFFLINE_METRICS_ONLY = flags.DEFINE_bool(
    'run_offline_metrics_only', False, 'Whether to run train or inference.'
)
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    'participant_module_path', None, 'Path to participant module'
)
_GIN_CONFIG = flags.DEFINE_string(
    'gin_config',
    None,
    'Path to gin config file that determines which experiment to run',
)

_EXTRA_GIN_BINDINGS = flags.DEFINE_multi_string(
    'extra_gin_bindings',
    [],
    'Extra gin bindings to add to the default bindings',
)
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to use')
_ALGOS = flags.DEFINE_list(
    'algos',
    ['ppo'],
    'Algorithms to run. If multiple are specified, they will be run in sequence', )
_EXPERIMENT_NUMBER = flags.DEFINE_string('experiment_number', None,
                                         'Experiment number')
_NUM_WEBSITES = flags.DEFINE_list('num_websites', None, 'Number of websites')
_MOTION_FILES = flags.DEFINE_list('motion_files', None, 'Motion files to run')
_SEEDS = flags.DEFINE_list('seeds', None, 'Seeds to run')
_MODE = flags.DEFINE_enum('mode', 'train', ['train', 'inference'],
                          'Mode of execution')
_SKILL_LEVELS = flags.DEFINE_list('skill_levels', None, 'Skill levels to run')

REPO_DIR = os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCKER_INSTRUCTIONS = {
    'quadruped_locomotion': [
        '''ARG APT_COMMAND="apt-get -o Acquire::Retries=3 \
          --no-install-recommends -y"''',
        'ENV DEBIAN_FRONTEND=noninteractive',
        '''
        RUN ${APT_COMMAND} update && \
          ${APT_COMMAND} install software-properties-common && \
          add-apt-repository ppa:deadsnakes/ppa && \
          ${APT_COMMAND} update && \
          ${APT_COMMAND} install \
          python3.9 \
          python3.9-dev \
          python3.9-venv \
          python3.9-distutils \
          wget \
          sudo \
          build-essential \
          libnss3 \
          unzip \
          x11-apps \
          libopenmpi-dev \
          x11-utils \
          && rm -rf /var/lib/apt/lists/*
        ''',

        # Add TensorRT
        '''
        RUN ${APT_COMMAND} update && \
          ${APT_COMMAND} install python3-libnvinfer-dev \
          python3-libnvinfer''',

        '''
        RUN wget https://bootstrap.pypa.io/get-pip.py && \
          python3.9 get-pip.py && \
          rm get-pip.py
        ''',
        '''
        RUN update-alternatives --install /usr/bin/python \
          python /usr/bin/python3.9 1 && update-alternatives --set python \
          /usr/bin/python3.9 \
          && rm -rf /var/lib/apt/lists/*
        ''',
        '''
        RUN echo "clouduser:password" | chpasswd && \
          echo "clouduser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
        ''',
        'WORKDIR /home/clouduser',
        '''
        RUN python3.9 -m venv venv && \
          . venv/bin/activate && \
          pip install --upgrade pip setuptools
        ''',
        'ENV PATH="/home/clouduser/venv/bin:${PATH}"',
        'RUN ln -s /home/clouduser/venv/bin/pip /usr/bin/pip',

        # Set up CUDA environment variables so that they appear BEFORE the virtualenv
        # This is necessary because we install tensorflow[and-cuda] in the virtualenv
        'ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"',
        'ENV CUDA_HOME="/usr/local/cuda-11.8"',
        'ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"',
        'ENV CUDNN_VERSION="8.7"',
        'ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"',

        'WORKDIR /workdir',
        f'''COPY {REPO_DIR}/a2perf/metrics/reliability/requirements.txt \
          ./a2perf/metrics/reliability/requirements.txt''',
        f'''
        COPY {REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt \
          ./a2perf/metrics/system/codecarbon/
        ''',
        f'''
        COPY {REPO_DIR}/a2perf/domains/quadruped_locomotion/requirements.txt \
          ./a2perf/domains/quadruped_locomotion/requirements.txt
        ''',
        f'''
        COPY {REPO_DIR}/a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt \
          ./a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt
        ''',
        f'COPY {REPO_DIR}/requirements.txt ./requirements.txt',
        'RUN pip install -r ./requirements.txt',
        'RUN pip install -r ./a2perf/domains/quadruped_locomotion/requirements.txt',
        'RUN pip install -r ./a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt',
        f'COPY {REPO_DIR} .',
        'RUN chmod -R 777 /workdir/a2perf /workdir/setup.py',
        'RUN pip install /workdir',
    ],
    'web_navigation': [
        '''ARG APT_COMMAND="apt-get -o Acquire::Retries=3 \
          --no-install-recommends -y"''',
        'ENV DEBIAN_FRONTEND=noninteractive',
        '''
        RUN ${APT_COMMAND} update && \
          ${APT_COMMAND} install software-properties-common && \
          add-apt-repository ppa:deadsnakes/ppa && \
          ${APT_COMMAND} update && \
          ${APT_COMMAND} install \
          python3.10 \
          python3.10-dev \
          python3.10-venv \
          python3.10-distutils \
          wget \
          sudo \
          build-essential \
          libnss3 \
          unzip \
          x11-apps \
          x11-utils \
          dbus  \
          dbus-x11 \
          libreadline-dev \
          less \
          && rm -rf /var/lib/apt/lists/*
        ''',

        # Add TensorRT
        '''
        RUN ${APT_COMMAND} update && \
          ${APT_COMMAND} install python3-libnvinfer-dev \
          python3-libnvinfer''',
        '''
        RUN wget https://bootstrap.pypa.io/get-pip.py && \
          python3.10 get-pip.py && \
          rm get-pip.py
        ''',
        '''
        RUN update-alternatives --install /usr/bin/python \
          python /usr/bin/python3.10 1 && update-alternatives --set python \
          /usr/bin/python3.10 \
          && rm -rf /var/lib/apt/lists/*
        ''',

        # Chrome Installation
        'ARG CHROME_VERSION="114.0.5735.90-1"',
        'ARG CHROMEDRIVER_VERSION="114.0.5735.90"',
        '''
        RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb && \
          ${APT_COMMAND} update && \
          ${APT_COMMAND} --fix-broken install && \
          ${APT_COMMAND} install /tmp/chrome.deb xvfb && \
          rm /tmp/chrome.deb
        ''',
        # [OPTIONAL] Some cloud images block internet access, so download chromedriver before experiment launch
        # webdriver-manager package expects the drivers to be in /home/user/.wdm folder. ('/home/user/.wdm/drivers/chromedriver/linux64/120.0.6099.109/chromedriver-linux64/chromedriver')
        # We also need to create a JSON entry for the driver like this
        # {
        #    "linux64_chromedriver_120.0.6099.109_for_120.0.6099": {
        #        "timestamp": "29/12/2023",
        #        "binary_path": "/home/user/.wdm/drivers/chromedriver/linux64/120.0.6099.109/chromedriver-linux64/chromedriver"
        #    }
        # }
        '''
        RUN TODAYS_DATE=$(date +%Y-%m-%d) && \
            wget --no-verbose -O /tmp/chromedriver_linux64.zip https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip && \
            unzip -o /tmp/chromedriver_linux64.zip -d /tmp/ && \
            mkdir -p /home/clouduser/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION} && \
            mv /tmp/chromedriver /home/clouduser/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION}/ && \
            rm /tmp/chromedriver_linux64.zip && \
            printf '{"linux64_chromedriver_%s_for_%s": {"timestamp": "%s", "binary_path": "/home/clouduser/.wdm/drivers/chromedriver/linux64/%s/chromedriver"}}' "${CHROMEDRIVER_VERSION}" "${CHROME_VERSION}" "${TODAYS_DATE}" "${CHROMEDRIVER_VERSION}" > /home/clouduser/.wdm/drivers.json && \
            chmod -R 777 /home/clouduser/.wdm
        ''',

        # Also add it to /root/.wdm
        '''RUN cp -r /home/clouduser/.wdm /root/''',
        '''
        RUN echo "clouduser:password" | chpasswd && \
          echo "clouduser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
        ''',
        'WORKDIR /home/clouduser',
        '''
        RUN python3.10 -m venv venv && \
          . venv/bin/activate && \
          pip install --upgrade pip setuptools
        ''',
        'ENV PATH="/home/clouduser/venv/bin:${PATH}"',
        'RUN ln -s /home/clouduser/venv/bin/pip /usr/bin/pip',

        # Set up CUDA environment variables so that they appear BEFORE the virtualenv
        # This is necessary because we install tensorflow[and-cuda] in the virtualenv
        'ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"',
        'ENV CUDA_HOME="/usr/local/cuda-11.8"',
        'ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"',
        'ENV CUDNN_VERSION="8.7"',
        'ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"',

        'WORKDIR /workdir',
        f'''COPY {REPO_DIR}/a2perf/metrics/reliability/requirements.txt \
          ./a2perf/metrics/reliability/requirements.txt''',
        f'''
        COPY {REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt \
          ./a2perf/metrics/system/codecarbon/
        ''',
        f'''
        COPY {REPO_DIR}/a2perf/domains/web_navigation/requirements.txt \
          ./a2perf/domains/web_navigation/requirements.txt
        ''',
        f'''
        COPY {REPO_DIR}/a2perf/domains/web_navigation/gwob/miniwob_plusplus/python/requirements.txt \
          ./a2perf/domains/web_navigation/gwob/miniwob_plusplus/python/requirements.txt 
        ''',
        f'''
        COPY {REPO_DIR}/a2perf/a2perf_benchmark_submission/web_navigation/ppo_lstm/requirements.txt \
          ./a2perf/a2perf_benchmark_submission/web_navigation/ppo_lstm/requirements.txt
        ''',
        f'COPY {REPO_DIR}/requirements.txt ./requirements.txt',
        'RUN pip install -r ./requirements.txt',
        'RUN pip install -r ./a2perf/domains/web_navigation/requirements.txt',
        'RUN pip install -r ./a2perf/a2perf_benchmark_submission/web_navigation/ppo_lstm/requirements.txt',
        f'COPY {REPO_DIR} .',
        'RUN chmod -R 777 /workdir/a2perf /workdir/setup.py',
        'RUN pip install /workdir',

    ],
    'circuit_training': []
}

ENTRYPOINT = {
    'quadruped_locomotion': xm.CommandList([
        'python3.9 -u /workdir/launch/entrypoint.py',
    ]),
    'web_navigation': xm.CommandList([
        'service dbus start',
        'python3.10 -u /workdir/launch/entrypoint.py'
    ]),

    'circuit_training': xm.CommandList([])
}

BASE_IMAGE = {
    'quadruped_locomotion': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04',
    'web_navigation': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04',
    'circuit_training': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04',
}
PYTHON_VERSION = {
    'quadruped_locomotion': '3.9',
    'web_navigation': '3.10',
    'circuit_training': '3.9',
}
ENV_VARS = {
    'quadruped_locomotion': {'WRAPT_DISABLE_EXTENSIONS': 'true',
                             'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                             'TF_GPU_THREAD_MODE': 'gpu_private',
                             'TF_USE_LEGACY_KERAS': '1'
                             },
    'web_navigation': {'WRAPT_DISABLE_EXTENSIONS': 'true',
                       'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                       'TF_GPU_THREAD_MODE': 'gpu_private',
                       'TF_USE_LEGACY_KERAS': '1'
                       },
    'circuit_training': {'WRAPT_DISABLE_EXTENSIONS': 'true'}
}


def create_experiment_name(hparams):
  """Creates an experiment name from a dictionary of hyperparameters."""
  return '_'.join(f"{key}_{hparams[key]}" for key in sorted(hparams.keys()) if
                  key in ['seed', 'domain', 'algo', 'task', 'skill_level'])


def get_next_experiment_number(host_dir_base):
  try:
    base_number = host_dir_base.rstrip('/')
    print(base_number)
    last_exp_num = \
      sorted([int(d) for d in os.listdir(host_dir_base) if d.isdigit()])[-1]
  except IndexError:
    return "0001"
  except FileNotFoundError:
    return "0001"
  return f"{last_exp_num + 1:04d}"


def get_hparam_sweeps(domain, algo, debug):
  if domain == 'quadruped_locomotion':
    general_hyperparameters = {
        'batch_size': [32],
        'eval_interval': [100],
        'log_interval': [100],
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
              'num_epochs': [1],
              'learning_rate': [3e-4],
              'entropy_regularization': [1e-4],
          },
          'sac': {
              'learning_rate': [3e-4],
              'rb_capacity': [100000],
          },
      }
    else:
      general_hyperparameters.update({
          'env_batch_size': [44],
          'total_env_steps': [200000000],
          'train_checkpoint_interval': [1000000],
          'policy_checkpoint_interval': [1000000],
          'timesteps_per_actorbatch': [4096],
      })

      algo_hyperparameters = {
          'ppo': {
              'entropy_regularization': [1e-4],
              'learning_rate': [3e-4],
              'num_epochs': [10],
          },
          'sac': {
              'learning_rate': [3e-4],
              'rb_capacity': [10000000],
          },
      }
  elif domain == 'web_navigation':
    general_hyperparameters = {
        'eval_interval': [100],
        'log_interval': [100],
    }

    if debug:
      general_hyperparameters.update({
          'env_batch_size': [4],
          'total_env_steps': [1000000],
          'train_checkpoint_interval': [10000],
          'policy_checkpoint_interval': [10000],
          'timesteps_per_actorbatch': [256],
      })

      algo_hyperparameters = {
          'ppo_lstm': {
              'algo': ['ppo_lstm'],
              'batch_size': [32],
              'num_epochs': [1],
              'learning_rate': [3e-4],
              'entropy_regularization': [1e-4],
          },
          'ddqn_lstm': {
              'algo': ['ddqn_lstm'],
              'batch_size': [512],
              'epsilon_greedy': [0.1],
              'learning_rate': [3e-4],
              'rb_capacity': [50000],
          },
      }
    else:
      general_hyperparameters.update({
          'env_batch_size': [8],
          'total_env_steps': [200000000],
          'train_checkpoint_interval': [1000000],
          'policy_checkpoint_interval': [1000000],
          'timesteps_per_actorbatch': [4096],
      })

      algo_hyperparameters = {
          'ppo_lstm': {
              'algo': ['ppo_lstm'],
              'batch_size': [32],
              'entropy_regularization': [1e-4],
              'learning_rate': [3e-4],
              'num_epochs': [10],
          },
          'ddqn_lstm': {
              'algo': ['ddqn_lstm'],
              'batch_size': [512],
              'learning_rate': [3e-4],
              'epsilon_greedy': [0.1],
              'rb_capacity': [10000000],
          },
      }
  elif domain == 'circuit_training':
    pass

  else:
    raise ValueError(f"Unknown domain: {domain}")

  # Combine general and algorithm-specific hyperparameters
  hyperparameters = {**general_hyperparameters,
                     **algo_hyperparameters[algo]}

  # Generate all combinations of hyperparameters
  keys, values = zip(*hyperparameters.items())
  hparam_sweeps = [dict(zip(keys, v)) for v in itertools.product(*values)]
  return hparam_sweeps


def main(_):
  with xm_local.create_experiment(
      experiment_title=_EXPERIMENT_NAME.value) as experiment:
    # Define Executor
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
            interactive=_INTERACTIVE.value, ),
        experimental_stream_output=True, )

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
        )])

    for algo in _ALGOS.value:
      hparam_sweeps = get_hparam_sweeps(domain=_DOMAIN.value, algo=algo,
                                        debug=_DEBUG.value)
      for i, hparam_config in enumerate(hparam_sweeps):
        for seed in _SEEDS.value:
          # Build the tasks depending on the domain
          tasks = []

          if _DOMAIN.value == 'quadruped_locomotion':
            tasks = _MOTION_FILES.value
          elif _DOMAIN.value == 'web_navigation':
            # permutation of num websites and difficulty level
            # so task will be of the form difficulty_0_num_websites_0 for example
            tasks = [f'difficulty_{difficulty}_num_websites_{num_websites}' for
                     difficulty in _DIFFICULTY_LEVELS.value for num_websites in
                     _NUM_WEBSITES.value]

          for task in tasks:
            skill_levels = _SKILL_LEVELS.value
            if not skill_levels:
              skill_levels = ['novice']

            for skill_level in skill_levels:
              dataset_id = f'{_DOMAIN.value[0].upper() + _DOMAIN.value[1:]}-{task}-{skill_level}-v0'
              hparam_config.update(dict(
                  seed=seed,
                  algo=algo,
                  task=task,
                  python_version=PYTHON_VERSION[_DOMAIN.value],
                  domain=_DOMAIN.value,
                  mode=_MODE.value,
                  dataset_id=dataset_id,
                  skill_level=skill_level,
                  gin_config=os.path.join('/workdir/a2perf/submission/configs',
                                          _DOMAIN.value,
                                          'debug' if _DEBUG.value else '',
                                          f'{_MODE.value}.gin'),
                  participant_module_path=os.path.join(
                      '/workdir/a2perf/a2perf_benchmark_submission',
                      _DOMAIN.value,
                      algo,
                  ),
                  run_offline_metrics_only=_RUN_OFFLINE_METRICS_ONLY.value,
              ))

              root_dir = os.path.join('/gcs',
                                      'a2perf',
                                      _DOMAIN.value,
                                      task,
                                      algo,
                                      'debug' if _DEBUG.value else '', )

              if _EXPERIMENT_NUMBER.value:
                experiment_number = _EXPERIMENT_NUMBER.value
              else:
                local_gcs_path = os.path.expanduser(_ROOT_DIR.value)
                local_root_dir = root_dir.replace('/gcs', local_gcs_path)
                experiment_number = get_next_experiment_number(local_root_dir)

              experiment_name = create_experiment_name(hparam_config)
              root_dir = os.path.join(root_dir, experiment_number,
                                      experiment_name)
              hparam_config['root_dir'] = root_dir

              if _DOMAIN.value == 'quadruped_locomotion':
                hparam_config.update(dict(
                    motion_file_path=os.path.join(
                        '/workdir/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/',
                        task + '.txt'), ))
              elif _DOMAIN.value == 'web_navigation':
                hparam_config.update(dict(use_xvfb=False,
                                          max_vocab_size=500,
                                          embedding_dim=100,
                                          latent_dim=50,
                                          profile_value_dropout=0.0,
                                          ))

                # TODO: Fix this cuz it doesn't launch all the tasks
                for difficulty_level in _DIFFICULTY_LEVELS.value:
                  for num_websites in _NUM_WEBSITES.value:
                    hparam_config.update(dict(

                        difficulty_level=difficulty_level,
                        num_websites=num_websites, ))

                    experiment.add(xm.Job(
                        args=hparam_config,
                        executable=executable,
                        executor=executor,
                    ))


if __name__ == '__main__':
  app.run(main)
