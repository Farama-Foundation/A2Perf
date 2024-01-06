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
_ALGOS = flags.DEFINE_list(
    'algos',
    ['ppo'],
    'Algorithms to run. If multiple are specified, they will be run in sequence', )
_EXPERIMENT_NUMBER = flags.DEFINE_string('experiment_number', None,
                                         'Experiment number')
_TASKS = flags.DEFINE_list('tasks', None, 'Tasks to run')
_SEEDS = flags.DEFINE_list('seeds', None, 'Seeds to run')
_MODE = flags.DEFINE_enum('mode', 'train', ['train', 'inference'],
                          'Mode of execution')
_SKILL_LEVELS = flags.DEFINE_list('skill_levels', None, 'Skill levels to run')

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
        'RUN ln -s /home/clouduser/venv/bin/pip /usr/bin/pip',
        'WORKDIR /workdir',
        f'COPY {REPO_DIR}/a2perf/metrics/reliability/requirements.txt ./a2perf/metrics/reliability/requirements.txt',
        f'COPY {REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt ./a2perf/metrics/system/codecarbon/',
        f'COPY {REPO_DIR}/a2perf/domains/quadruped_locomotion/requirements.txt ./a2perf/domains/quadruped_locomotion/requirements.txt',
        f'COPY {REPO_DIR}/a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt ./a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt',
        f'COPY {REPO_DIR}/requirements.txt ./requirements.txt',
        'RUN pip install -r ./requirements.txt',
        'RUN pip install -r ./a2perf/domains/quadruped_locomotion/requirements.txt',
        'RUN pip install -r ./a2perf/a2perf_benchmark_submission/quadruped_locomotion/ppo/requirements.txt',
        f'COPY {REPO_DIR} .',
        'RUN chmod -R 777 /workdir/a2perf /workdir/setup.py',
        'RUN pip install /workdir'
    ],
    'web_navigation': [],
    'circuit_training': []
}

ENTRYPOINT = {
    'quadruped_locomotion': xm.CommandList([
        'sudo service ssh start',
        'sudo service dbus start',
        'python3.9 /workdir/launch/entrypoints/quadruped_locomotion.py',
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


def get_hparam_sweeps(domain, debug):
  if domain == 'quadruped_locomotion':
    if debug:
      # Debug mode: simpler hyperparameters for faster iteration
      hyperparameters = {
          'batch_size': [512],
          'num_epochs': [20],
          'env_batch_size': [32],
          'total_env_steps': [1000000],
          'learning_rate': [3e-4],
          'eval_interval': [10000],
          'train_checkpoint_interval': [100000],
          'policy_checkpoint_interval': [100000],
          'log_interval': [10000],
          'entropy_regularization': [0.05],
          'timesteps_per_actorbatch': [1024]
      }
    else:
      # Normal mode: more extensive range of hyperparameters
      hyperparameters = {
          'batch_size': [32],
          'num_epochs': [20],
          'env_batch_size': [40],
          'total_env_steps': [1000000],
          'learning_rate': [3e-4],
          'eval_interval': [100000],
          'train_checkpoint_interval': [10000],
          'policy_checkpoint_interval': [10000],
          'log_interval': [1000],
          'entropy_regularization': [0.05],
          'timesteps_per_actorbatch': [4096]
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
  with xm_local.create_experiment(
      experiment_title=_EXPERIMENT_NAME.value) as experiment:
    hparam_sweeps = get_hparam_sweeps(_DOMAIN.value, _DEBUG.value)

    # Define Executor
    executor = xm_local.Local(
        requirements=xm.JobRequirements(
            resources={
                xm.ResourceType.LOCAL_GPU: 8,
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
        ),
    ])

    for i, hparam_config in enumerate(hparam_sweeps):
      for seed in _SEEDS.value:
        for algo in _ALGOS.value:
          for task in _TASKS.value:
            skill_levels = _SKILL_LEVELS.value
            if not skill_levels:
              skill_levels = ['novice']

            for skill_level in skill_levels:
              dataset_id = f'{_DOMAIN.value[0].upper() + _DOMAIN.value[1:]}-{task}-{skill_level}-v0'
              hparam_config.update(dict(
                  seed=seed,
                  algo=algo,
                  task=task,
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
              hparam_config.update(dict(
                  root_dir=root_dir,
              ))

              if _DOMAIN.value == 'quadruped_locomotion':
                hparam_config.update(dict(
                    motion_file_path=os.path.join(
                        '/workdir/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/',
                        task + '.txt'), ))

              experiment.add(xm.Job(
                  args=hparam_config,
                  executable=executable,
                  executor=executor,
              ))


if __name__ == '__main__':
  app.run(main)
