import os
import subprocess

from absl import app
from absl import flags
from absl import logging
from pyfiglet import Figlet
from termcolor import colored

_EXPERIMENT_ID = flags.DEFINE_string(
    'experiment_id', None, 'Experiment ID for web navigation.'
)
_EXPERIMENT_IDS = flags.DEFINE_list(
    'experiment_ids',
    None,
    'List of experiment ids to use for generating datasets.',
)
_NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations', None, 'Number of iterations for training.'
)
_NUM_REPLICAS = flags.DEFINE_integer(
    'num_replicas', None, 'Number of replicas for distributed training.'
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
_VOCABULARY_MANAGER_AUTH_KEY = flags.DEFINE_string(
    'vocabulary_manager_auth_key',
    None,
    'Authentication key for the manager server.',
)

_JOB_TYPE = flags.DEFINE_enum(
    'job_type',
    None,
    ['train', 'collect', 'inference', 'evaluate', 'generate'],
    'Type of job',
)
_NUM_EVAL_EPISODES = flags.DEFINE_integer(
    'num_eval_episodes', 100, 'Number of episodes to evaluate the policy.'
)
_NUM_EPISODES_TO_GENERATE = flags.DEFINE_integer(
    'num_episodes_to_generate', 0, 'Number of episodes to generate datasets.'
)
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

_NUM_EPISODES_PER_ITERATION = flags.DEFINE_integer(
    'num_episodes_per_iteration', -1, 'Number of episodes per iteration.'
)
_MAX_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'max_sequence_length', -1, 'Max sequence length.'
)
_NUM_COLLECT_MACHINES = flags.DEFINE_integer(
    'num_collect_machines',
    -1,
    'Number of machines used to generate the dataset.',
)
_NUM_COLLECT_JOBS_PER_MACHINE = flags.DEFINE_integer(
    'num_collect_jobs_per_machine', None, 'Number of collect jobs per machine.'
)
_TRAIN_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'train_checkpoint_interval', None, 'Train checkpoint interval.'
)
_POLICY_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'policy_checkpoint_interval', None, 'Policy checkpoint interval.'
)
_POLICY_NAME = flags.DEFINE_string('policy_name', None, 'Policy name.')
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

_VOCABULARY_SERVER_PORT = flags.DEFINE_integer(
    'vocabulary_server_port', None, 'Vocabulary server port.'
)
_VOCABULARY_SERVER_ADDRESS = flags.DEFINE_string(
    'vocabulary_server_address', None, 'Address for the vocabulary manager.'
)
_NETLIST_PATH = flags.DEFINE_string(
    'netlist_path', None, 'Path to the netlist file.'
)
_INIT_PLACEMENT_PATH = flags.DEFINE_string(
    'init_placement_path', None, 'Path to the initial placement file.'
)
_STD_CELL_PLACER_MODE = flags.DEFINE_enum(
    'std_cell_placer_mode',
    None,
    ['dreamplace', 'fd'],
    'Mode for the standard cell placer.',
)

_MODE = flags.DEFINE_enum(
    'mode',
    None,
    ['train', 'collect', 'reverb', 'inference', 'evaluate', 'generate'],
    'Mode.',
)
_TASK_NAME = flags.DEFINE_string('task_name', None, 'Name of the task.')

_DATASETS_PATH = flags.DEFINE_string(
    'datasets_path', None, 'Path to save the dataset to.'
)


def main(_):
  os.environ['TASK_NAME'] = _TASK_NAME.value
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
  os.environ['TRAIN_CHECKPOINT_INTERVAL'] = str(
      _TRAIN_CHECKPOINT_INTERVAL.value
  )
  os.environ['POLICY_CHECKPOINT_INTERVAL'] = str(
      _POLICY_CHECKPOINT_INTERVAL.value
  )
  os.environ['EVAL_INTERVAL'] = str(_EVAL_INTERVAL.value)
  os.environ['LOG_INTERVAL'] = str(_LOG_INTERVAL.value)
  os.environ['DATASET_ID'] = _DATASET_ID.value
  os.environ['MINARI_DATASETS_PATH'] = _DATASETS_PATH.value
  os.environ['DEBUG'] = str(_DEBUG.value)
  os.environ['NUM_EPISODES_PER_ITERATION'] = str(
      _NUM_EPISODES_PER_ITERATION.value
  )

  # Export distributed training variables
  os.environ['NUM_ITERATIONS'] = str(_NUM_ITERATIONS.value)
  os.environ['NUM_COLLECT_JOBS_PER_MACHINE'] = str(
      _NUM_COLLECT_JOBS_PER_MACHINE.value
  )
  os.environ['MAX_SEQUENCE_LENGTH'] = str(_MAX_SEQUENCE_LENGTH.value)
  os.environ['NUM_REPLICAS'] = str(_NUM_REPLICAS.value)
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
  os.environ['VOCABULARY_MANAGER_AUTH_KEY'] = _VOCABULARY_MANAGER_AUTH_KEY.value

  # For collect, change the root dir to a subdirectory to make sure that the
  # training system metrics are not overwritten
  figlet_obj = Figlet(font='standard', width=300)
  if _JOB_TYPE.value == 'collect':
    # Change the root dir to the machine's hostname
    root_dir = os.path.join(
        _ROOT_DIR.value, _JOB_TYPE.value, os.environ.get('HOSTNAME', 'unknown')
    )
    print(f'Changing root dir to {root_dir}')
    print(colored(figlet_obj.renderText(_JOB_TYPE.value), 'red'))
  elif _JOB_TYPE.value == 'inference' and _MODE.value == 'inference':
    # We leave the root dir alone since we need to load the model produced
    # by the training job
    root_dir = _ROOT_DIR.value
    print(colored(figlet_obj.renderText(_JOB_TYPE.value), 'red'))

    os.environ['POLICY_NAME'] = _POLICY_NAME.value
    logging.info('Performing inference with the %s policy.', _POLICY_NAME.value)
  elif _JOB_TYPE.value == 'train':
    print(f'Experiment ID: {_EXPERIMENT_ID.value}')
    root_dir = _ROOT_DIR.value
    print(colored(figlet_obj.renderText('Copy this'), 'red'))
    print('Experiment ID: ', _EXPERIMENT_ID.value)
  elif _JOB_TYPE.value == 'evaluate':
    # If the job type is `evaluate`, then we do not need to run the
    # benchmarking flow at all. This pathway is used for evaluating
    # ALL policies in a given training run. We can use these returns to classify
    # the policies into intermediate, novice, and expert.
    root_dir = _ROOT_DIR.value

    os.environ['POLICY_NAME'] = _POLICY_NAME.value
    logging.info('Evaluating the policy %s.', _POLICY_NAME.value)
  elif _JOB_TYPE.value == 'generate':
    # For dataset generation, we want the root dir to be at
    # the domain level, so we can load the expertise data
    # os.path.join(a2perf, _DOMAIN.value, str(exp_id), task_name, algo, exp_name
    #         work_unit_id,
    #     )
    # so the root dir should be at the domain level
    root_dir = os.path.join(_ROOT_DIR.value, *(['..'] * 5))
    root_dir = os.path.abspath(root_dir)

    os.environ['POLICY_NAME'] = _POLICY_NAME.value
    logging.info(
        'Performing dataset generation with %s policy.', _POLICY_NAME.value
    )
  else:
    raise ValueError(f'Invalid job type: {_JOB_TYPE.value}')

  os.environ['ROOT_DIR'] = root_dir
  os.environ['ALGO'] = _ALGO.value
  os.environ['ENV_NAME'] = _ENV_NAME.value
  if _ALGO.value == 'sac':
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  elif _ALGO.value == 'ddqn':
    os.environ['EPSILON_GREEDY'] = str(_EPSILON_GREEDY.value)
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  elif _ALGO.value == 'ppo':
    os.environ['ENTROPY_REGULARIZATION'] = str(_ENTROPY_REGULARIZATION.value)
    os.environ['NUM_EPOCHS'] = str(_NUM_EPOCHS.value)
    os.environ['USE_GAE'] = str(_USE_GAE.value)
    os.environ['RB_CAPACITY'] = '100000'
  elif _ALGO.value == 'td3':
    os.environ['EXPLORATION_NOISE_STD'] = str(_EXPLORATION_NOISE_STD.value)
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)
  elif _ALGO.value == 'ddpg':
    os.environ['RB_CAPACITY'] = str(_RB_CAPACITY.value)

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
    os.environ['NETLIST_PATH'] = _NETLIST_PATH.value
    os.environ['INIT_PLACEMENT_PATH'] = _INIT_PLACEMENT_PATH.value
    os.environ['STD_CELL_PLACER_MODE'] = _STD_CELL_PLACER_MODE.value
  else:
    raise ValueError(f'Invalid domain in entrypoint.py: {_DOMAIN.value}')

  if _JOB_TYPE.value == 'evaluate':
    command = [
        'python',
        '-m',
        'a2perf.analysis.evaluation',
        f'--num_eval_episodes={_NUM_EVAL_EPISODES.value}',
        f'--root_dir={root_dir}',
        f'--env_name={_ENV_NAME.value}',
        f'--max_parallel_envs={_NUM_COLLECT_JOBS_PER_MACHINE.value}',
        f'--verbosity={logging.get_verbosity()}',
    ]
  elif _JOB_TYPE.value == 'generate':
    # Before generating datasets, we must use the evaluation data to classify
    # the policies into novice, intermediate, and expert. We can then use these
    # classifications to generate datasets.

    # Only the leading worker determines skill level
    job_completion_index = int(os.environ.get('JOB_COMPLETION_INDEX', -1))
    is_leading_worker = job_completion_index == 0
    if is_leading_worker:
      skill_level_command = [
          'python',
          '-m',
          'a2perf.analysis.expertise',
          f'--root_dir={root_dir}',
          f'--verbosity={logging.get_verbosity()}',
          '--average_measure=median',
          f'--experiment_ids={",".join(_EXPERIMENT_IDS.value)}',
          f'--task_name={_TASK_NAME.value}',
          f'--skill_level={_SKILL_LEVEL.value}',
      ]

      print(skill_level_command)
      skill_level_process = subprocess.Popen(
          skill_level_command, env=os.environ.copy(), text=True
      )
      skill_level_process.wait()
      if skill_level_process.returncode != 0:
        raise ValueError(f'Error running the command: {skill_level_command}')
      else:
        print('Finished running the command successfully.')

    generate_command = [
        'python',
        '-m',
        'a2perf.data.generate',
        f'--env_name={_ENV_NAME.value}',
        f'--root_dir={root_dir}',
        f'--verbosity={logging.get_verbosity()}',
        f'--num_episodes={_NUM_EPISODES_TO_GENERATE.value}',
        f'--num_processes={_NUM_COLLECT_JOBS_PER_MACHINE.value}',
        f'--skill_level={_SKILL_LEVEL.value}',
        f'--task_name={_TASK_NAME.value}',
        f'--seed={_SEED.value}',
        f'--datasets_path={root_dir}',
        f'--policy_name={_POLICY_NAME.value}',
        f'--num_machines={_NUM_COLLECT_MACHINES.value}',
        f'--replica_id={job_completion_index}',
    ]

    print(generate_command)
    subprocess.run(
        generate_command, env=os.environ.copy(), text=True, check=True
    )

    return
  else:
    command = [
        'python',
        '-m',
        'a2perf.submission.main_submission',
        f'--mode={_MODE.value}',
        f'--gin_config={_GIN_CONFIG.value}',
        f'--participant_module_path={_PARTICIPANT_MODULE_PATH.value}',
        f'--root_dir={root_dir}',
        f'--metric_values_dir={root_dir}/metrics',
        f'--run_offline_metrics_only={_RUN_OFFLINE_METRICS_ONLY.value}',
        f'--verbosity={logging.get_verbosity()}',
    ]

  if _USE_XVFB.value:
    command = ['xvfb-run'] + command

  process = subprocess.Popen(command, env=os.environ.copy(), text=True)

  process.wait()
  if process.returncode != 0:
    raise ValueError(f'Error running the command: {command}')
  else:
    print('Finished running the command successfully.')


if __name__ == '__main__':
  flags.mark_flags_as_required(
      [
          _DOMAIN.name,
          _ALGO.name,
          _SKILL_LEVEL.name,
          _MODE.name,
      ],
  )
  app.run(main)
