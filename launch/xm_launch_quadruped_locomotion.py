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
_ALGO = flags.DEFINE_string(
    'algo',
    None,
    'Name of algorithm to run',
)
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed')
_EXPERIMENT_NUMBER = flags.DEFINE_string('experiment_number', None,
                                         'Experiment number')
_MOTION_FILE_PATH = flags.DEFINE_string('motion_file_path',
                                        None,
                                        'Motion file')
_TASK = flags.DEFINE_string('task', None, 'Task')
_MODE = flags.DEFINE_string('mode', None, 'Mode to run in')
_SKILL_LEVEL = flags.DEFINE_string('skill_level', None, 'Skill level')

REPO_DIR = os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCKER_INSTRUCTIONS = {
    'quadruped_locomotion': [
        'ARG APT_COMMAND="apt-get -o Acquire::Retries=3 --no-install-recommends -y"',
        'ENV DEBIAN_FRONTEND=noninteractive',
        'RUN ${APT_COMMAND} update && ' +
        '${APT_COMMAND} install software-properties-common && ' +
        'add-apt-repository ppa:deadsnakes/ppa && ' +
        '${APT_COMMAND} update && ' +
        '${APT_COMMAND} upgrade && ' +
        '${APT_COMMAND} install --allow-change-held-packages ' +
        'python3.9 ' +
        'python3.9-dev ' +
        'python3.9-venv ' +
        'python3.9-distutils ' +
        'wget ' +
        'sudo ' +
        'build-essential ' +
        'ssh ' +
        'openssh-server ' +
        'libnss3 ' +
        'unzip ' +
        'x11-apps ' +
        'libopenmpi-dev ' +
        'x11-utils ' +
        'libcudnn8 ' +
        'libcudnn8-dev && ' +
        'rm -rf /var/lib/apt/lists/*',
        'RUN wget https://bootstrap.pypa.io/get-pip.py && ' +
        'python3.9 get-pip.py && ' +
        'rm get-pip.py',
        'RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && ' +
        'update-alternatives --set python /usr/bin/python3.9',
        'RUN echo "X11UseLocalhost no\\nX11DisplayOffset 10\\nPasswordAuthentication yes\\nPort 2020" >> /etc/ssh/sshd_config && ' +
        'mkdir /run/sshd',
        'EXPOSE 2020',
        f'RUN groupadd -g {os.getgid()} user_group && ' +
        'groupadd -g 998 docker && ' +
        'usermod -aG sudo,user_group clouduser && ' +
        'echo "clouduser:password" | chpasswd && ' +
        'echo "clouduser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && ' +
        'mkdir -p /home/clouduser/.ssh && ' +
        'chmod 700 /home/clouduser/.ssh && ' +
        'touch /home/clouduser/.ssh/authorized_keys && ' +
        'chmod 600 /home/clouduser/.ssh/authorized_keys',
        'WORKDIR /home/clouduser',
        'RUN python3.9 -m venv venv && ' +
        '. venv/bin/activate && ' +
        'pip install --upgrade pip setuptools ',
        'ENV PATH="/home/clouduser/venv/bin:${PATH}"',
        'ENV PATH="/usr/local/cuda/bin:${PATH}"',
        'ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"',
        'WORKDIR /workdir',
        f'COPY {REPO_DIR}/a2perf/metrics/reliability/requirements.txt ./a2perf/metrics/reliability/requirements.txt',
        f'COPY {REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt ./a2perf/metrics/system/codecarbon/',
        f'COPY {REPO_DIR}/a2perf/domains/quadruped_locomotion/requirements.txt ./a2perf/domains/quadruped_locomotion/requirements.txt',
        f'COPY {REPO_DIR}/a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt ./a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt',
        f'COPY {REPO_DIR}/requirements.txt ./requirements.txt',
        'RUN pip install -r ./requirements.txt',
        'RUN pip install -r ./a2perf/domains/quadruped_locomotion/requirements.txt',
        f'COPY {REPO_DIR} .',
        'RUN chmod -R 777 .',
        'RUN pip install .',
    ],
    'web_navigation': [],
    'circuit_training': []
}

ENTRYPOINT = {
    'quadruped_locomotion': xm.CommandList([
        # make sure the terminal is kept open
        'sudo service ssh start && sudo service ssh status &&  bash'
    ]),
    'web_navigation': xm.CommandList([]),
    'circuit_training': xm.CommandList([])
}

BASE_IMAGE = {
    'quadruped_locomotion': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04',
    'web_navigation': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04',
    'circuit_training': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04'
}

ENV_VARS = {
    'quadruped_locomotion': {'WRAPT_DISABLE_EXTENSIONS': 'true'},
    'web_navigation': {'WRAPT_DISABLE_EXTENSIONS': 'true'},
    'circuit_training': {'WRAPT_DISABLE_EXTENSIONS': 'true'}
}


def create_experiment_name(hparams):
  """Creates an experiment name from a dictionary of hyperparameters."""
  return '_'.join(f"{key}_{hparams[key]}" for key in sorted(hparams.keys()) if
                  key in ['seed', 'domain', 'algo', 'task', 'skill_level'])


def get_hparam_sweeps(domain, debug):
  if domain == 'quadruped_locomotion':
    if debug:
      # Debug mode: simpler hyperparameters for faster iteration
      hyperparameters = {
          'batch_size_values': [32],
          'num_epoch_values': [20],
          'env_batch_sizes': [8],
          'total_env_steps': [200000],
          'learning_rates': [3e-4],
          'eval_intervals': [1000],
          'train_checkpoint_intervals': [5000],
          'policy_checkpoint_intervals': [5000],
          'log_intervals': [100],
          'entropy_regularization_values': [0.05],
          'timesteps_per_actorbatch_values': [512]
      }
    else:
      # Normal mode: more extensive range of hyperparameters
      hyperparameters = {
          'batch_size_values': [512],
          'num_epoch_values': [10],
          'env_batch_sizes': [32],
          'total_env_steps': [200000000],
          'learning_rates': [3e-4],
          'eval_intervals': [100000],
          'train_checkpoint_intervals': [100000],
          'policy_checkpoint_intervals': [100000],
          'log_intervals': [10000],
          'entropy_regularization_values': [0.05],
          'timesteps_per_actorbatch_values': [4096]
      }

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())
    hparam_sweeps = [dict(zip(keys, v)) for v in itertools.product(*values)]
  elif domain == 'web_navigation':
    pass
  else:
    raise ValueError(f"Unknown domain: {domain}")
  return hparam_sweeps


def main(_):
  # set directory of this script as working directory
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # If experiment number is defined, replace the last part of root_dir with experiment number
  if _EXPERIMENT_NUMBER.value is not None:
    root_dir_flag = os.path.join(os.path.dirname(_ROOT_DIR.value),
                                 _EXPERIMENT_NUMBER.value)
  else:
    root_dir_flag = _ROOT_DIR.value

  with xm_local.create_experiment(
      experiment_title=_EXPERIMENT_NAME.value) as experiment:
    hparam_sweeps = get_hparam_sweeps(_DOMAIN.value, _DEBUG.value)

    # Define Executor
    executor = xm_local.Local(docker_options=xm_local.DockerOptions(
        ports=None,
        volumes=None,
        mount_gcs_path=False,
        interactive=_INTERACTIVE.value, ),
        experimental_stream_output=True, )

    # Define Executable
    [executable] = experiment.package([
        xm.python_container(
            executor_spec=xm_local.LocalSpec(),
            path='../',
            use_deep_module=True,
            base_image=BASE_IMAGE[_DOMAIN.value],
            docker_instructions=DOCKER_INSTRUCTIONS[_DOMAIN.value],
            entrypoint=ENTRYPOINT[_DOMAIN.value],
            env_vars=ENV_VARS[_DOMAIN.value],
        ),
    ])

    for i, hparam_config in enumerate(hparam_sweeps):
      experiment_name = create_experiment_name(hparam_config)
      root_dir = os.path.abspath(root_dir_flag)
      root_dir = os.path.join(root_dir, experiment_name)

      if _SKILL_LEVEL.value is not None:
        dataset_id = f'{_DOMAIN.value[0].upper() + _DOMAIN.value[1:]}-{_TASK.value}-{_SKILL_LEVEL.value}-v0'
      else:
        dataset_id = None

      # Add additional arguments that are constant across all runs
      hparam_config.update(dict(
          dataset_id=dataset_id,
          domain=_DOMAIN.value,
          seed=_SEED.value,
          algo=_ALGO.value,
          task=_TASK.value,
          skill_level=_SKILL_LEVEL.value,
          extra_gin_bindings=','.join(_EXTRA_GIN_BINDINGS.value),
          gin_config=_GIN_CONFIG.value,
          mode=_MODE.value,
          motion_file_path=_MOTION_FILE_PATH.value,
          participant_module_path=_PARTICIPANT_MODULE_PATH.value,
          root_dir=root_dir,
          debug=str(_DEBUG.value).lower(),
          run_offline_metrics_only=_RUN_OFFLINE_METRICS_ONLY,
          train_logs_dirs=','.join(_TRAIN_LOGS_DIRS.value),
      ))

      print(hparam_config)
      hparam_config.pop('domain')
      hparam_config.pop('task')
      hparam_config.pop('algo')

      # Export all hyperparameters as environment variables
      env_vars = {}
      for key, value in hparam_config.items():
        env_vars[key.upper()] = str(value)

      hparam_config.clear()
      experiment.add(xm.Job(
          args=hparam_config,
          env_vars=env_vars,
          executable=executable,
          executor=executor,
      ))


if __name__ == '__main__':
  app.run(main)
