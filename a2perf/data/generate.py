import itertools
import multiprocessing
import os
import shutil
from typing import Any
from typing import OrderedDict
from typing import Union

from a2perf.domains import circuit_training
from a2perf.domains import quadruped_locomotion
from a2perf.domains import web_navigation
from absl import app
from absl import flags
from absl import logging
import gym as legacy_gym
import gymnasium as gym
from gymnasium import spaces
import minari
from minari import DataCollector
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.policies import policy_loader
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import time_step as ts

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory of the environment. If not set, the ROOT_DIR environment '
    'variable is used.',
)
_NUM_EPISODES = flags.DEFINE_integer(
    'num_episodes', 100, 'Number of episodes to evaluate the policy.'
)
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', 1, 'Number of processes to use.'
)
_TASK_NAME = flags.DEFINE_string(
    'task_name',
    'evaluation',
    'Name of the task to perform. This is used to name the dataset.',
)
_SEED = flags.DEFINE_integer('seed', 0, 'Seed to use.')
_DATASETS_PATH = flags.DEFINE_string(
    'datasets_path',
    '/mnt/gcs/a2perf/datasets/quadruped_locomotion',
    'Path to save the dataset to.',
)
_AUTHOR = flags.DEFINE_string('author', 'Ikechukwu Uchendu', 'Author name.')
_AUTHOR_EMAIL = flags.DEFINE_string(
    'author_email', 'iuchendu@g.harvard.edu', 'Author email.'
)
_CODE_PERMALINK = flags.DEFINE_string('code_permalink', '', 'Code permalink.')
_SKILL_LEVEL = flags.DEFINE_enum(
    'skill_level',
    'novice',
    ['novice', 'intermediate', 'expert'],
    'Skill level of the expert.',
)

_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment.')


def delete_dataset_wrapper(unique_id):
  unique_id = f'{unique_id:03d}'
  dataset_path = os.path.join(
      os.path.expanduser(_DATASETS_PATH.value),
      _TASK_NAME.value,
      _SKILL_LEVEL.value,
      unique_id,
  )
  shutil.rmtree(dataset_path)


def create_domain(env_name) -> gym.Env:
  if env_name == 'CircuitTraining-v0':
    netlist_path = os.environ.get('NETLIST_PATH', None)
    init_placement_path = os.environ.get('INIT_PLACEMENT_PATH', None)
    kwargs = {
        'netlist_file': netlist_path,
        'init_placement': init_placement_path,
    }
  elif env_name == 'QuadrupedLocomotion-v0':
    motion_file_path = os.environ.get('MOTION_FILE_PATH', None)
    num_parallel_envs = os.environ.get('NUM_PARALLEL_ENVS', 1)
    kwargs = {
        'motion_files': [motion_file_path],
        'num_parallel_envs': num_parallel_envs,
    }
  elif env_name == 'WebNavigation-v0':
    kwargs = {}
  else:
    raise ValueError(f'Unknown environment: {env_name}')

  return gym.make(env_name, **kwargs)


def load_policy(saved_model_path: str, checkpoint_path: str) -> TFPolicy:
  """Loads a policy model from the environment's root directory.

  Args:
      saved_model_path: The path of a directory containing a full saved model.
      checkpoint_path: The path to a directory that contains variable
        checkpoints (as opposed to full saved models) for the policy.

  Returns:
      The loaded policy.
  """
  policy = policy_loader.load(
      saved_model_path=saved_model_path,
      checkpoint_path=checkpoint_path,
  )
  return policy


def preprocess_observation(
    observation: Union[np.ndarray, Any],
    reward: float = 0.0,
    discount: float = 1.0,
    step_type: ts.StepType = ts.StepType.MID,
    time_step_spec: ts.TimeStep = None,
) -> ts.TimeStep:
  """Preprocesses a raw observation from the Gym environment into a TF Agents TimeStep.

  Args:
      observation: Raw observation from the environment.
      reward: The reward received after the last action.
      discount: The discount factor.
      step_type: The type of the current step.
      time_step_spec: The spec of the time_step used to extract dtype and shape.

  Returns:
      A preprocessed TimeStep object suitable for the policy.
  """
  if isinstance(observation, (dict, OrderedDict)):
    processed_observation = {}
    for key, value in observation.items():
      if time_step_spec and key in time_step_spec.observation.keys():
        spec = time_step_spec.observation[key]
        # Adjust dtype and shape according to the time_step_spec
        processed_observation[key] = tf.convert_to_tensor(
            value, dtype=spec.dtype
        )
      else:
        # Use the numpy dtype of the element that was passed in
        processed_observation[key] = tf.convert_to_tensor(
            value, dtype=value.dtype
        )
    observation = processed_observation
  elif isinstance(observation, np.ndarray):
    if time_step_spec:
      shape = time_step_spec.observation.shape
      observation = tf.convert_to_tensor(
          observation, dtype=time_step_spec.observation.dtype
      )
      observation.set_shape(shape)
    else:
      # Convert the ndarray directly, using its own dtype
      observation = tf.convert_to_tensor(observation, dtype=observation.dtype)
  else:
    raise ValueError(f'Unknown observation type: {type(observation)}')

  # Convert step_type, reward, and discount using their respective dtypes from time_step_spec
  # if it is provided, otherwise default to the dtype inferred from the input
  step_type = tf.convert_to_tensor(
      step_type,
      dtype=time_step_spec.step_type.dtype
      if time_step_spec
      else step_type.dtype,
  )
  reward = tf.convert_to_tensor(
      reward,
      dtype=time_step_spec.reward.dtype if time_step_spec else np.float32,
  )
  discount = tf.convert_to_tensor(
      discount,
      dtype=time_step_spec.discount.dtype if time_step_spec else np.float32,
  )

  return ts.TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation,
  )


def perform_rollouts(
    policy: TFPolicy, env: gym.Env, num_episodes: int
) -> np.ndarray:
  """Perform rollouts using the policy and environment.

  Args:
      policy: The policy to use.
      env: The environment to use.
      num_episodes: The number of episodes to perform.

  Returns:
      The returns for each episode.
  """

  episode_returns = []
  for _ in range(num_episodes):
    episode_return = 0
    terminated = truncated = False
    obs, info = env.reset()
    while not terminated and not truncated:
      obs = preprocess_observation(obs, time_step_spec=policy.time_step_spec)
      action_step = policy.action(obs)
      obs, reward, terminated, truncated, info = env.step(action_step.action)
      episode_return += reward
    episode_returns.append(episode_return)

  return np.array(episode_returns)


def collect_dataset(
    env_name,
    checkpoint_path,
    unique_id,
    num_episodes,
    seed,
):
  unique_id = f'{unique_id:03d}'
  dataset_path = os.path.join(
      os.path.expanduser(_DATASETS_PATH.value),
      _TASK_NAME.value,
      _SKILL_LEVEL.value,
      str(unique_id),
  )
  os.environ['MINARI_DATASETS_PATH'] = dataset_path
  np.random.seed(seed)
  tf.random.set_seed(seed)

  env = create_domain(env_name=env_name)

  if env_name == 'QuadrupedLocomotion-v0':
    infinite_observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=env.observation_space.shape,
        dtype=env.observation_space.dtype,
    )
    data_collector_env = DataCollector(
        env,
        observation_space=infinite_observation_space,
        action_space=env.action_space,
    )
  else:
    data_collector_env = DataCollector(
        env,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

  saved_model_path = os.path.join(checkpoint_path, '..', '..', 'policy')
  policy = load_policy(
      saved_model_path=saved_model_path, checkpoint_path=checkpoint_path
  )

  _ = perform_rollouts(policy, data_collector_env, num_episodes)

  # To make a temp dataset id, insert the unique_id right after the -v0
  # in the supplied dataset id
  temp_dataset_id = f'{_ENV_NAME.value[:-3]}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-{unique_id}-v0'
  dataset = data_collector_env.create_dataset(
      dataset_id=temp_dataset_id,
  )
  env.close()
  return dataset


def main(_):
  if _DATASETS_PATH.value is not None:
    os.environ['MINARI_DATASETS_PATH'] = os.path.expanduser(
        _DATASETS_PATH.value
    )

  root_dir = os.path.expanduser(_ROOT_DIR.value)
  env_name = _ENV_NAME.value[:-3]
  dataset_id = f'{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-v0'
  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)

  # Load the dataframe containing evaluation data
  evaluation_df = pd.read_csv(
      os.path.join(root_dir, 'evaluation_data_with_skill_levels.csv')
  )

  # Filter the dataframe by the skill level
  evaluation_df = evaluation_df[evaluation_df.skill_level == _SKILL_LEVEL.value]
  if evaluation_df.empty:
    logging.warning('No policies found for skill level: %s', _SKILL_LEVEL.value)
    return

  # Divide up the number of samples to collect between the policies
  num_episodes_per_policy = np.ceil(
      _NUM_EPISODES.value / len(evaluation_df)
  ).astype(int)
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    tasks = list(
        zip(
            itertools.repeat(_ENV_NAME.value),
            evaluation_df['checkpoint_path'],
            range(len(evaluation_df)),
            itertools.repeat(num_episodes_per_policy),
            itertools.repeat(_SEED.value),
        )
    )
    datasets = pool.starmap(collect_dataset, tasks)
    pool.close()
    pool.join()
  logging.info('Finished collecting all episodes.')

  logging.info('Combining datasets...')
  dataset = minari.combine_datasets(
      datasets_to_combine=datasets, new_dataset_id=dataset_id
  )

  logging.info('Successfully combined datasets')
  logging.info('\tTotal steps: %s', dataset.total_steps)
  logging.info('\tTotal episodes: %s', dataset.total_episodes)

  logging.info('Cleaning up temporary datasets.')
  with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
    pool.map(delete_dataset_wrapper, range(len(datasets)))
    pool.close()
    pool.join()
  logging.info(f'Finished cleaning up temporary datasets.')


if __name__ == '__main__':
  app.run(main)
