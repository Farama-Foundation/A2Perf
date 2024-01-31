import os
import select
import subprocess

from absl import app
from absl import flags
from pyfiglet import Figlet
from termcolor import colored

_EXPERIMENT_ID = flags.DEFINE_string(
    'experiment_id', None, 'Experiment ID for web navigation.'
)
_NUM_COLLECT_MACHINES = flags.DEFINE_integer(
    'num_collect_machines', None, 'Number of machines for collecting data.'
)
_MAX_VOCAB_SIZE = flags.DEFINE_integer(
    'max_vocab_size', None, 'Max vocab size for web navigation.'
)
_LATENT_DIM = flags.DEFINE_integer(
    'latent_dim', None, 'Latent dimension for web navigation.'
)
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', None, 'Embedding dimension for web navigation.'
)
_PROFILE_VALUE_DROPOUT = flags.DEFINE_float(
    'profile_value_dropout', None, 'Profile value dropout for web navigation.'
)

_DEBUG = flags.DEFINE_bool('debug', False, 'Debugging mode')
_DOMAIN = flags.DEFINE_enum(
    'domain',
    None,
    ['quadruped_locomotion', 'web_navigation', 'circuit_training'],
    'Domain to run.',
)
_SEED = flags.DEFINE_integer('seed', None, 'Global seed.')
_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', None, 'Number of environments to run in parallel.'
)
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment.')
_EPSILON_GREEDY = flags.DEFINE_float(
    'epsilon_greedy', None, 'Epsilon greedy value.'
)
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', None, 'Number of epochs.')
_TOTAL_ENV_STEPS = flags.DEFINE_integer(
    'total_env_steps', None, 'Total steps in the environment.'
)
_USE_XVFB = flags.DEFINE_boolean('use_xvfb', False, 'Use xvfb.')
_USE_GAE = flags.DEFINE_boolean('use_gae', False, 'Use GAE.')
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
_AUTH_KEY = flags.DEFINE_string(
    'auth_key', 'secretkey', 'Authentication key for the manager server.'
)

_JOB_TYPE = flags.DEFINE_enum(
    'job_type', None, ['train', 'collect', 'reverb'], 'Type of job'
)
# _TASK = flags.DEFINE_string('task', None, 'Task to run.')
_GIN_CONFIG = flags.DEFINE_string('gin_config', None, 'Gin config file.')
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
_EXPLORATION_NOISE_STD = flags.DEFINE_float(
    'exploration_noise_std', None, 'Exploration noise std.'
)
_DATASET_ID = flags.DEFINE_string('dataset_id', None, 'Dataset ID.')

_NUM_WEBSITES = flags.DEFINE_integer(
    'num_websites', None, 'Number of websites for web navigation.'
)
_DIFFICULTY_LEVEL = flags.DEFINE_enum(
    'difficulty_level',
    None,
    ['1', '2', '3'],
    'Difficulty level for web navigation.',
)

_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_REPLAY_BUFFER_SERVER_PORT = flags.DEFINE_integer(
    'replay_buffer_server_port', None, 'Replay buffer server port.'
)
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
)

_VARIABLE_CONTAINER_SERVER_PORT = flags.DEFINE_integer(
    'variable_container_server_port', None, 'Variable container server port.'
)

_VOCABULARY_SERVER_ADDRESS = flags.DEFINE_string(
    'vocabulary_server_address', None, 'Vocabulary server address.'
)
_VOCABULARY_SERVER_PORT = flags.DEFINE_integer(
    'vocabulary_server_port', None, 'Vocabulary server port.'
)

_MODE = flags.DEFINE_enum(
    'mode', None, ['train', 'collect', 'reverb', 'inference'], 'Mode.'
)


def main(_):
  os.environ['EXPERIMENT_ID'] = _EXPERIMENT_ID.value
  os.environ['SEED'] = str(_SEED.value)
  os.environ['ENV_BATCH_SIZE'] = str(_ENV_BATCH_SIZE.value)
  os.environ['TOTAL_ENV_STEPS'] = str(_TOTAL_ENV_STEPS.value)
  os.environ['ALGORITHM'] = _ALGO.value
  os.environ['SKILL_LEVEL'] = _SKILL_LEVEL.value
  os.environ['GIN_CONFIG'] = _GIN_CONFIG.value
  os.environ['LEARNING_RATE'] = str(_LEARNING_RATE.value)
  os.environ['BATCH_SIZE'] = str(_BATCH_SIZE.value)
  os.environ['PARTICIPANT_MODULE_PATH'] = _PARTICIPANT_MODULE_PATH.value
  os.environ['RUN_OFFLINE_METRICS_ONLY'] = str(_RUN_OFFLINE_METRICS_ONLY.value)
  os.environ['TIMESTEPS_PER_ACTOR_BATCH'] = str(_TIMESTEPS_PER_ACTORBATCH.value)
  os.environ['TRAIN_CHECKPOINT_INTERVAL'] = str(
      _TRAIN_CHECKPOINT_INTERVAL.value
  )
  os.environ['POLICY_CHECKPOINT_INTERVAL'] = str(
      _POLICY_CHECKPOINT_INTERVAL.value
  )
  os.environ['NUM_COLLECT_MACHINES'] = str(_NUM_COLLECT_MACHINES.value)
  os.environ['EVAL_INTERVAL'] = str(_EVAL_INTERVAL.value)
  os.environ['LOG_INTERVAL'] = str(_LOG_INTERVAL.value)
  os.environ['DATASET_ID'] = _DATASET_ID.value
  os.environ['DEBUG'] = str(_DEBUG.value)
  os.environ['TIMESTEPS_PER_ACTORBATCH'] = str(_TIMESTEPS_PER_ACTORBATCH.value)

  # Export distributed training variables
  os.environ['MODE'] = _MODE.value
  os.environ['JOB_TYPE'] = str(_JOB_TYPE.value)
  os.environ['REPLAY_BUFFER_SERVER_ADDRESS'] = (
      _REPLAY_BUFFER_SERVER_ADDRESS.value
  )
  os.environ['REPLAY_BUFFER_SERVER_PORT'] = str(
      _REPLAY_BUFFER_SERVER_PORT.value
  )
  os.environ['VARIABLE_CONTAINER_SERVER_ADDRESS'] = (
      _VARIABLE_CONTAINER_SERVER_ADDRESS.value
  )
  os.environ['VARIABLE_CONTAINER_SERVER_PORT'] = str(
      _VARIABLE_CONTAINER_SERVER_PORT.value
  )
  os.environ['VOCABULARY_SERVER_ADDRESS'] = _VOCABULARY_SERVER_ADDRESS.value
  os.environ['VOCABULARY_SERVER_PORT'] = str(_VOCABULARY_SERVER_PORT.value)
  os.environ['AUTH_KEY'] = _AUTH_KEY.value
  # For collect/inference, change the root dir to a subdirectory to make sure
  # That our system metrics are not overwritten
  if _JOB_TYPE.value in ['collect', 'inference']:
    # Change the root dir to the machine's hostname
    root_dir = os.path.join(
        _ROOT_DIR.value, _JOB_TYPE.value, os.environ.get('HOSTNAME', 'unknown')
    )
    print(f'Changing root dir to {root_dir}')
    f = Figlet(font='standard', width=300)
    print(colored(f.renderText(_JOB_TYPE.value), 'red'))

  else:
    print(f'Experiment ID: {_EXPERIMENT_ID.value}')
    root_dir = _ROOT_DIR.value
    f = Figlet(font='standard')
    print(colored(f.renderText('Copy this'), 'red'))
    print('Experiment ID: ', _EXPERIMENT_ID.value)
  os.environ['ROOT_DIR'] = root_dir

  os.environ['ALGO'] = _ALGO.value
  if _ALGO.value == 'sac':
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  elif _ALGO.value == 'ddqn':
    os.environ['EPSILON_GREEDY'] = str(_EPSILON_GREEDY.value)
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  elif _ALGO.value == 'ppo':
    os.environ['ENTROPY_REGULARIZATION'] = str(_ENTROPY_REGULARIZATION.value)
    os.environ['NUM_EPOCHS'] = str(_NUM_EPOCHS.value)
    os.environ['USE_GAE'] = str(_USE_GAE.value)
    os.environ['RB_CAPACITY'] = '10000000'  # doesn't matter for PPO
  elif _ALGO.value == 'td3':
    os.environ['EXPLORATION_NOISE_STD'] = str(_EXPLORATION_NOISE_STD.value)
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  elif _ALGO.value == 'ddpg':
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  os.environ['ENV_NAME'] = _ENV_NAME.value
  if _DOMAIN.value == 'quadruped_locomotion':
    os.environ['DOMAIN'] = 'quadruped_locomotion'
    os.environ['MOTION_FILE_PATH'] = _MOTION_FILE_PATH.value
  elif _DOMAIN.value == 'web_navigation':
    os.environ['DIFFICULTY_LEVEL'] = _DIFFICULTY_LEVEL.value
    os.environ['DOMAIN'] = 'web_navigation'
    os.environ['EMBEDDING_DIM'] = str(_EMBEDDING_DIM.value)
    os.environ['LATENT_DIM'] = str(_LATENT_DIM.value)
    os.environ['MAX_VOCAB_SIZE'] = str(_MAX_VOCAB_SIZE.value)
    os.environ['NUM_WEBSITES'] = str(_NUM_WEBSITES.value)
    os.environ['PROFILE_VALUE_DROPOUT'] = str(_PROFILE_VALUE_DROPOUT.value)
  elif _DOMAIN.value == 'circuit_training':
    os.environ['DOMAIN'] = 'circuit_training'
  else:
    raise ValueError(f'Invalid domain in entrypoint.py: {_DOMAIN.value}')

  command = [
      'python',
      'a2perf/submission/main_submission.py',
      f'--gin_config={_GIN_CONFIG.value}',
      f'--participant_module_path={_PARTICIPANT_MODULE_PATH.value}',
      f'--root_dir={root_dir}',
      f'--metric_values_dir={root_dir}/metrics',
      f'--run_offline_metrics_only={_RUN_OFFLINE_METRICS_ONLY.value}',
  ]

  if _USE_XVFB.value:
    command = ['xvfb-run'] + command
  process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env=os.environ.copy(),
  )

  while True:
    if process.poll() is not None:
      break
    # Check if there's data to read
    readable, _, _ = select.select([process.stdout], [], [], 0.1)
    if readable:
      output = process.stdout.readline()
      if output:
        print(output.strip().decode('utf-8', 'ignore'))


if __name__ == '__main__':
  flags.mark_flags_as_required(
      [
          _DOMAIN.name,
          _ALGO.name,
          _SKILL_LEVEL.name,
      ],
  )
  app.run(main)
