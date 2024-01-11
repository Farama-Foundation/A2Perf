import os
import subprocess

from absl import app
from absl import flags

_DEBUG = flags.DEFINE_bool('debug', False, 'Debugging mode')
_SEED = flags.DEFINE_integer('seed', None, 'Global seed.')
_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', None, 'Number of environments to run in parallel.'
)
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', None, 'Number of epochs.')
_TOTAL_ENV_STEPS = flags.DEFINE_integer(
    'total_env_steps', None, 'Total steps in the environment.'
)
_MOTION_FILE_PATH = flags.DEFINE_string(
    'motion_file_path', None, 'Motion file path.'
)
_ALGO = flags.DEFINE_string('algo', None, 'Algorithm used.')
_SKILL_LEVEL = flags.DEFINE_enum(
    'skill_level',
    None,
    ['novice', 'intermediate', 'expert'],
    'Skill level of the expert.',
)
_TASK = flags.DEFINE_enum(
    'task', None, ['dog_pace', 'dog_trot', 'dog_spin'], 'Task to run.'
)
_GIN_CONFIG = flags.DEFINE_string('gin_config', None, 'Gin config file.')
_MODE = flags.DEFINE_string('mode', None, 'Mode of execution.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', None, 'Learning rate.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', None, 'Batch size for training.'
)
_ROOT_DIR = flags.DEFINE_string('root_dir', None, 'Root directory.')
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    'participant_module_path', None, 'Participant module path.'
)
_RUN_OFFLINE_METRICS_ONLY = flags.DEFINE_boolean(
    'run_offline_metrics_only', None, 'Run offline metrics only.'
)
_RB_CAPACITY = flags.DEFINE_integer(
    'rb_capacity', None, 'Replay buffer capacity.'
)

_TIMESTEPS_PER_ACTORBATCH = flags.DEFINE_integer(
    'timesteps_per_actorbatch', None, 'Timesteps per actor batch.'
)
_TRAIN_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'train_checkpoint_interval', None, 'Train checkpoint interval.'
)
_POLICY_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'policy_checkpoint_interval', None, 'Policy checkpoint interval.'
)
_EVAL_INTERVAL = flags.DEFINE_integer('eval_interval', None, 'Eval interval.')
_LOG_INTERVAL = flags.DEFINE_integer('log_interval', None, 'Log interval.')

_ENTROPY_REGULARIZATION = flags.DEFINE_float(
    'entropy_regularization', None, 'Entropy regularization.'
)
_DATASET_ID = flags.DEFINE_string('dataset_id', None, 'Dataset ID.')


def main(_):
  os.environ['SEED'] = str(_SEED.value)
  os.environ['ENV_BATCH_SIZE'] = str(_ENV_BATCH_SIZE.value)
  os.environ['NUM_EPOCHS'] = str(_NUM_EPOCHS.value)
  os.environ['TOTAL_ENV_STEPS'] = str(_TOTAL_ENV_STEPS.value)
  os.environ['MOTION_FILE_PATH'] = _MOTION_FILE_PATH.value
  os.environ['ALGO'] = _ALGO.value
  os.environ['SKILL_LEVEL'] = _SKILL_LEVEL.value
  os.environ['TASK'] = _TASK.value
  os.environ['GIN_CONFIG'] = _GIN_CONFIG.value
  os.environ['MODE'] = _MODE.value
  os.environ['LEARNING_RATE'] = str(_LEARNING_RATE.value)
  os.environ['BATCH_SIZE'] = str(_BATCH_SIZE.value)
  os.environ['ROOT_DIR'] = _ROOT_DIR.value
  os.environ['PARTICIPANT_MODULE_PATH'] = _PARTICIPANT_MODULE_PATH.value
  os.environ['RUN_OFFLINE_METRICS_ONLY'] = str(_RUN_OFFLINE_METRICS_ONLY.value)
  os.environ['TIMESTEPS_PER_ACTOR_BATCH'] = str(_TIMESTEPS_PER_ACTORBATCH.value)
  os.environ['TRAIN_CHECKPOINT_INTERVAL'] = str(
      _TRAIN_CHECKPOINT_INTERVAL.value
  )
  os.environ['POLICY_CHECKPOINT_INTERVAL'] = str(
      _POLICY_CHECKPOINT_INTERVAL.value
  )
  os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  os.environ['EVAL_INTERVAL'] = str(_EVAL_INTERVAL.value)
  os.environ['LOG_INTERVAL'] = str(_LOG_INTERVAL.value)
  os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
  os.environ['TF_USE_LEGACY_KERAS'] = '1'
  os.environ['ENTROPY_REGULARIZATION'] = str(_ENTROPY_REGULARIZATION.value)
  os.environ['DATASET_ID'] = _DATASET_ID.value
  os.environ['DEBUG'] = str(_DEBUG.value)
  os.environ['TIMESTEPS_PER_ACTORBATCH'] = str(_TIMESTEPS_PER_ACTORBATCH.value)
  command = (
      'python3.9 a2perf/submission/main_submission.py '
      f'--gin_config={_GIN_CONFIG.value} '
      f'--participant_module_path={_PARTICIPANT_MODULE_PATH.value} '
      f'--root_dir={_ROOT_DIR.value} '
      f'--run_offline_metrics_only={_RUN_OFFLINE_METRICS_ONLY.value}'
  )

  process = subprocess.Popen(
      command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
  )
  while True:
    output = process.stdout.readline()
    if process.poll() is not None:
      break
    if output:
      print(output.strip().decode('utf-8', 'ignore'))


if __name__ == '__main__':
  app.run(main)
