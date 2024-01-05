import copy
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
_TRAIN_LOGS_DIRS = flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train'],
    'Directory patterns fr train logs that will be used to calculate reliability metrics. Should be glob patterns',
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

FLAGS = flags.FLAGS


def main(_):
  # set directory of this script as working directory
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # If experiment number is defined, replace the last part of root_dir with experiment number
  if _EXPERIMENT_NUMBER.value is not None:
    root_dir_flag = os.path.join(os.path.dirname(_ROOT_DIR.value),
                                 _EXPERIMENT_NUMBER.value)
  else:
    root_dir_flag = _ROOT_DIR.value
  quadruped_locomotion_dir = os.path.join(os.getcwd(),
                                          '../a2perf/domains/quadruped_locomotion')
  if _LOCAL.value:
    executable_path = '/usr/bin/bash'
    binary_path = './local/quadruped_locomotion/launch.sh'
    additional_args = []
    env_vars = dict(
        quadruped_locomotion_DIR=quadruped_locomotion_dir,
        TF_FORCE_GPU_ALLOW_GROWTH='true',
        DISPLAY=os.environ.get('DISPLAY', ''),

    )
  else:
    # Create log dirs since singularity needs them to exist
    executable_path = '/usr/bin/sbatch'
    binary_path = './singularity/quadruped_locomotion/launch.slurm'
    additional_args = []
    env_vars = dict(
        TF_FORCE_GPU_ALLOW_GROWTH='true',
        DISPLAY=os.environ.get('DISPLAY', ''),
        # TF_GPU_ALLOCATOR='cuda_malloc_async' # doesn't work on some of the FASRC machines???
    )

  with xm_local.create_experiment(
      experiment_title=_EXPERIMENT_NAME.value) as experiment:
    if _DEBUG.value:
      quadruped_locomotion_seeds = [
          _SEED.value,
      ]
      batch_size_values = [32]
      num_epoch_values = [20]
      env_batch_sizes = [3]
      total_env_steps = [200000, ]
      int_save_freqs = [100000]
      int_eval_freqs = [10000]
      learning_rates = [3e-4]
      algos = [_ALGO.value]
    else:
      quadruped_locomotion_seeds = [
          _SEED.value,
      ]
      batch_size_values = [512]
      num_epoch_values = [500]
      total_env_steps = [200000000, ]
      env_batch_sizes = [3]
      int_save_freqs = [1000000]
      int_eval_freqs = [100000]
      algos = [_ALGO.value]
      learning_rates = [3e-4]

    quadruped_locomotion_hparam_sweeps = list(
        dict([
            ('seed', seed),
            ('total_env_steps', env_steps),
            ('env_batch_size', env_batch_size),
            ('int_save_freq', int_save_freq),
            ('algo', algo),
            ('int_eval_freq', int_eval_freq),
            ('batch_size', batch_size),
            ('num_epochs', num_epochs),
            ('learning_rate', learning_rate),
        ])
        for
        seed, env_steps, env_batch_size, int_save_freq, algo, int_eval_freq, batch_size, num_epochs, learning_rate
        in
        itertools.product(quadruped_locomotion_seeds,
                          total_env_steps, env_batch_sizes,
                          int_save_freqs, algos, int_eval_freqs,
                          batch_size_values, num_epoch_values, learning_rates
                          )
    )

    # Define Executable
    [executable] = experiment.package([
        xm.binary(
            path=executable_path,
            args=[binary_path] + additional_args,
            executor_spec=xm_local.LocalSpec(),
            env_vars=env_vars,
        )
    ])

    for hparam_config in quadruped_locomotion_hparam_sweeps:
      hparam_reduced = dict()
      for key in hparam_config.keys():
        new_key = ''.join(
            [x[0] for x in key.split('_')])
        hparam_reduced[new_key] = hparam_config[key]

      hparam_reduced = hparam_config  # replacing for debug
      experiment_name = _EXPERIMENT_NAME.value + '_' + '_'.join(
          f"{key}_{hparam_reduced[key]}" for key in
          sorted(hparam_reduced.keys()))

      root_dir = os.path.abspath(root_dir_flag)
      root_dir = os.path.join(root_dir, experiment_name)
      participant_module_path = os.path.join(_PARTICIPANT_MODULE_PATH.value)
      run_offline_metrics_only = str(_RUN_OFFLINE_METRICS_ONLY.value)

      if _SKILL_LEVEL.value is not None:
        dataset_id = f'QuadrupedLocomotion-{_TASK.value}-{_SKILL_LEVEL.value}-v0'
      else:
        dataset_id = None

      # Add additional arguments that are constant across all runs
      hparam_config.update(dict(
          dataset_id=dataset_id,
          extra_gin_bindings=','.join(_EXTRA_GIN_BINDINGS.value),
          gin_config=_GIN_CONFIG.value,
          mode=_MODE.value,
          motion_file_path=_MOTION_FILE_PATH.value,
          participant_module_path=participant_module_path,
          quad_loco_dir=quadruped_locomotion_dir,
          root_dir=root_dir,
          debug=str(_DEBUG.value).lower(),
          run_offline_metrics_only=run_offline_metrics_only,
          skill_level=_SKILL_LEVEL.value,
          train_logs_dirs=','.join(_TRAIN_LOGS_DIRS.value),
      ))

      print(hparam_config)
      experiment.add(xm.Job(
          args=hparam_config,
          env_vars=env_vars,
          executable=executable,
          executor=xm_local.Local(),
      ))


if __name__ == '__main__':
  app.run(main)
