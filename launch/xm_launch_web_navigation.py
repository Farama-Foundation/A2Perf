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
_SKILL_LEVEL = flags.DEFINE_enum('skill_level', None,
                                 ['novice', 'intermediate', 'expert'],
                                 'Skill level')
_TASK = flags.DEFINE_enum('task', None, ['1', '2', '3'],
                          'Task')
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


def create_experiment_name(hparams):
  """Creates an experiment name from a dictionary of hyperparameters."""
  return '_'.join(f"{key}_{hparams[key]}" for key in sorted(hparams.keys()) if
                  key in ['seed', 'domain', 'algo', 'task', 'skill_level'])


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
      log_intervals = [1000]
      rb_capacity_values = [100000, ]
      rb_checkpoint_intervals = [
          5000]
      batch_size_values = [32, ]
      timesteps_per_actorbatch_values = [256]
      web_nav_seeds = [_SEED.value]
      epsilon_greedy_values = [0.2]
      env_batch_sizes = [3]
      total_env_steps = [100000]
      learning_rates = [1e-4]
      eval_intervals = [1000]
      train_checkpoint_intervals = [5000]
      policy_checkpoint_intervals = [5000]
      entropy_regularization_values = [0.01]
    else:
      epsilon_greedy_values = [0.1]
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
      learning_rates = [1e-4]
      eval_intervals = [50000]
      train_checkpoint_intervals = [100000]
      policy_checkpoint_intervals = [100000]
      entropy_regularization_values = [0.01]

    web_nav_hparam_sweeps = [
        {
            'epsilon_greedy': epsilon_greedy,
            'seed': seed,
            'env_batch_size': env_batch_size,
            'total_env_steps': env_steps,
            'learning_rate': learning_rate,
            'eval_interval': eval_interval,
            'entropy_regularization': er,
            'train_checkpoint_interval': train_ci,
            'policy_checkpoint_interval': policy_ci,
            'log_interval': li,
            'rb_capacity': rb,
            'rb_checkpoint_interval': rci,  # Added rb_checkpoint_interval
            'batch_size': bs,
            'timesteps_per_actorbatch': tpab,
        }
        for
        seed, env_batch_size, env_steps, learning_rate, eval_interval,
        train_ci, policy_ci, li, rb, rci, bs, tpab, epsilon_greedy, er
        in
        itertools.product(
            web_nav_seeds,
            env_batch_sizes,
            total_env_steps,
            learning_rates,
            eval_intervals,
            train_checkpoint_intervals,
            policy_checkpoint_intervals,
            log_intervals,
            rb_capacity_values,
            rb_checkpoint_intervals,
            batch_size_values,
            timesteps_per_actorbatch_values,
            epsilon_greedy_values,
            entropy_regularization_values,
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
      hparam_config.update(dict(
          domain='quadruped_locomotion',
          algo=_ALGO.value,
          task=_TASK.value,
          difficulty_level=_DIFFICULTY_LEVEL.value,
          num_websites=_NUM_WEBSITES.value,
          skill_level=_SKILL_LEVEL.value,
      ))

      experiment_name = create_experiment_name(hparam_config)
      hparam_config.pop('domain')
      hparam_config.pop('task')
      root_dir = os.path.abspath(root_dir_flag)
      root_dir = os.path.join(root_dir, experiment_name)
      participant_module_path = os.path.join(_PARTICIPANT_MODULE_PATH.value)
      run_offline_metrics_only = str(_RUN_OFFLINE_METRICS_ONLY.value)

      if _SKILL_LEVEL.value is not None:
        dataset_id = f'WebNavigation-difficulty_level_{_TASK.value}-{_SKILL_LEVEL.value}-v0'
      else:
        dataset_id = None

      hparam_config.update(dict(root_dir=root_dir,
                                dataset_id=dataset_id,
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
