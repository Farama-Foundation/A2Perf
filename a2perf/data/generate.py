import functools
import itertools
import multiprocessing
import os
import shutil

import gymnasium as gym
import minari
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from minari import DataCollector

from a2perf.domains.tfa import suite_gym
from a2perf.domains.tfa.utils import load_policy
from a2perf.domains.tfa.utils import perform_rollouts

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
_POLICY_NAME = flags.DEFINE_string('policy_name', None, 'Name of the policy.')


def delete_dataset_wrapper(unique_id):
  unique_id = f'{unique_id:03d}'
  dataset_path = os.path.join(
      os.path.expanduser(_DATASETS_PATH.value),
      _TASK_NAME.value,
      _SKILL_LEVEL.value,
      unique_id,
  )
  shutil.rmtree(dataset_path)


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

  # We need to hack the observation space for quadruped_locomotion
  if env_name == 'QuadrupedLocomotion-v0':
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(160,), dtype=np.float64
    )
    data_collector_call = functools.partial(
        DataCollector, observation_space=observation_space
    )
  else:
    data_collector_call = DataCollector

  env = suite_gym.create_domain(
      env_name=env_name,
      root_dir=_ROOT_DIR.value,
      gym_env_wrappers=[data_collector_call],
  )

  # checkpoint_path = checkpoint_path.replace('/gcs', os.path.expanduser('~/gcs'))
  saved_model_path = os.path.join(
      checkpoint_path, '..', '..', _POLICY_NAME.value
  )
  policy = load_policy(
      saved_model_path=saved_model_path,
      checkpoint_path=checkpoint_path,
  )

  _ = perform_rollouts(policy, env, num_episodes)

  temp_dataset_id = f'{_ENV_NAME.value[:-3]}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-{unique_id}-v0'
  dataset = env.gym.create_dataset(
      dataset_id=temp_dataset_id,
  )
  env.close()
  return dataset


def main(_):
  if _DATASETS_PATH.value is not None:
    minari_datasets_path = os.path.join(
        os.path.expanduser(
            _DATASETS_PATH.value,
        ),
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
    )
    os.environ['MINARI_DATASETS_PATH'] = minari_datasets_path

  root_dir = os.path.expanduser(_ROOT_DIR.value)
  env_name = _ENV_NAME.value[:-3]
  dataset_id = f'{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-v0'
  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)

  # Load the dataframe containing evaluation data
  evaluation_df = pd.read_csv(
      os.path.join(
          root_dir,
          _TASK_NAME.value,
          _SKILL_LEVEL.value,
          'evaluation_data_with_skill_levels.csv',
      )
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
