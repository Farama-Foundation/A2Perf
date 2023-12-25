import copy
import itertools
import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', 'web_navigation', 'Name of experiment'
)
_EXPERIMENT_NUMBER = flags.DEFINE_string('experiment_number', None,
                                         'Experiment number')
_ROOT_DIR = flags.DEFINE_string(
    'root_dir', '/tmp/xm_local', 'Base directory for logs and results'
)
_TRAIN_LOGS_DIRS = flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train'],
    'Directory patterns fr train logs that will be used to calculate reliability metrics. Should be glob patterns',
)
_NUM_WEBSITES = flags.DEFINE_integer('num_websites', 1,
                                     'Number of websites to run')
_LOCAL = flags.DEFINE_bool('local', False, 'Run locally or on cluster')
_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
_ALGO = flags.DEFINE_string(
    'algo',
    None,
    'Name of algorithm to run',
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

_DIFFICULTY_LEVEL = flags.DEFINE_integer('difficulty_level', 1,
                                         'Difficulty level')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed')
FLAGS = flags.FLAGS


def main(_):
  # set directory of this script as working directory
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  if _EXPERIMENT_NUMBER.value is not None:
    root_dir_flag = os.path.join(os.path.dirname(_ROOT_DIR.value),
                                 _EXPERIMENT_NUMBER.value)
  else:
    root_dir_flag = _ROOT_DIR.value

  web_nav_dir = os.path.join(os.getcwd(), '../a2perf/domains/web_navigation')
  if _LOCAL.value:
    executable_path = '/usr/bin/bash'
    binary_path = 'local/web_navigation/launch.sh'
    additional_args = []
    env_vars = dict(
        WEB_NAV_DIR=web_nav_dir,
        TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_GPU_ALLOCATOR='cuda_malloc_async',
        DISPLAY=os.environ.get('DISPLAY', ''),
    )

  else:
    # Create log dirs since singularity needs them to exist
    executable_path = '/usr/bin/sbatch'
    binary_path = './singularity/web_navigation/launch.slurm'
    additional_args = []
    env_vars = dict(
    )

  with xm_local.create_experiment(
      experiment_title=_EXPERIMENT_NAME.value) as experiment:

    if _DEBUG.value:
      # Debug mode hyperparameters
      summary_intervals = [1000]
      log_intervals = [1000]
      rb_capacity_values = [10000, ]
      rb_checkpoint_intervals = [
          5000]  # Assuming a default value for debug mode
      batch_size_values = [32, ]
      timesteps_per_actorbatch_values = [256]
      web_nav_seeds = [_SEED.value]
      env_batch_sizes = [3]
      total_env_steps = [100000]
      difficulty_levels = [_DIFFICULTY_LEVEL.value]
      learning_rates = [1e-4]
      eval_intervals = [1000]
      train_checkpoint_intervals = [5000]
      policy_checkpoint_intervals = [5000]
      num_website_values = [_NUM_WEBSITES.value]
    else:
      # Non-debug mode hyperparameters
      algorithms = [_ALGO.value]
      summary_intervals = [50000]  # Adjusted to match the non-debug scale
      log_intervals = [50000]  # Adjusted to match the non-debug scale
      rb_capacity_values = [100000,
                            200000]  # Hypothetical values for non-debug mode
      rb_checkpoint_intervals = [
          5000]  # Assuming a default value for debug mode
      batch_size_values = [64, 128,
                           256]  # Hypothetical larger batch sizes for non-debug mode
      timesteps_per_actorbatch_values = [
          20]  # Hypothetical value for non-debug mode
      web_nav_seeds = [_SEED.value]
      env_batch_sizes = [16]
      total_env_steps = [1000000]
      difficulty_levels = [_DIFFICULTY_LEVEL.value]
      learning_rates = [1e-4]
      eval_intervals = [50000]
      train_checkpoint_intervals = [100000]
      policy_checkpoint_intervals = [100000]
      num_website_values = [_NUM_WEBSITES.value]
    web_nav_hparam_sweeps = [
        {
            'num_websites': num_websites,
            'seed': seed,
            'env_batch_size': env_batch_size,
            'total_env_steps': env_steps,
            'difficulty_level': difficulty_level,
            'learning_rate': lr,
            'eval_interval': ei,
            'train_checkpoint_interval': tci,
            'policy_checkpoint_interval': pci,
            'summary_interval': si,
            'log_interval': li,
            'rb_capacity': rb,
            'rb_checkpoint_interval': rci,  # Added rb_checkpoint_interval
            'batch_size': bs,
            'timesteps_per_actorbatch': tpab,
        }
        for
        seed, env_batch_size, env_steps, difficulty_level, lr, ei, tci, pci, si, li, rb, rci, bs, tpab, num_websites
        in
        itertools.product(
            web_nav_seeds,
            env_batch_sizes,
            total_env_steps,
            difficulty_levels,
            learning_rates,
            eval_intervals,
            train_checkpoint_intervals,
            policy_checkpoint_intervals,
            summary_intervals,
            log_intervals,
            rb_capacity_values,
            rb_checkpoint_intervals,  # Added rb_checkpoint_intervals here
            batch_size_values,
            timesteps_per_actorbatch_values,
            num_website_values
        )
    ]
    # Define Executable
    [executable] = experiment.package([
        xm.binary(
            path=executable_path,
            args=[binary_path] + additional_args,
            executor_spec=xm_local.LocalSpec(),
            env_vars=env_vars,
        )
    ])

    for i, hparam_config in enumerate(web_nav_hparam_sweeps):
      hparam_reduced = dict()
      for key in hparam_config.keys():
        new_key = ''.join(
            [x[0] for x in key.split('_')])
        hparam_reduced[new_key] = hparam_config[key]
      experiment_name = _EXPERIMENT_NAME.value + '_' + '_'.join(
          f"{key}_{hparam_reduced[key]}" for key in
          sorted(hparam_reduced.keys()))

      root_dir = os.path.abspath(root_dir_flag)
      root_dir = os.path.join(root_dir, experiment_name)
      participant_module_path = os.path.join(_PARTICIPANT_MODULE_PATH.value)
      run_offline_metrics_only = str(_RUN_OFFLINE_METRICS_ONLY.value)

      hparam_config.update(dict(root_dir=root_dir,
                                gin_config=_GIN_CONFIG.value,
                                participant_module_path=participant_module_path,
                                debug=str(_DEBUG.value).lower(),
                                web_nav_dir=web_nav_dir,
                                algo=_ALGO.value,
                                train_logs_dirs=_TRAIN_LOGS_DIRS.value,
                                run_offline_metrics_only=run_offline_metrics_only, ))

      print(hparam_config)
      experiment.add(xm.Job(
          executable=executable,
          executor=xm_local.Local(),
          args=hparam_config,
          env_vars=env_vars,
      ))


if __name__ == '__main__':
  app.run(main)
