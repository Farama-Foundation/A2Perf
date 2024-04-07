import functools
import itertools
import multiprocessing
import os
import shutil
import subprocess
import time

from a2perf.domains.tfa.utils import load_policy
from a2perf.domains.tfa.utils import perform_rollouts
from a2perf.domains.utils import suite_gym
from absl import app
from absl import flags
from absl import logging
import gymnasium as gym
import minari
from minari import DataCollector
import numpy as np
import pandas as pd
import tensorflow as tf

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory of the environment. If not set, the ROOT_DIR environment '
    'variable is used.',
)
_NUM_EPISODES = flags.DEFINE_integer(
    'num_episodes', None, 'Number of episodes to evaluate the policy.'
)
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', None, 'Number of processes to use.'
)
_REPLICA_ID = flags.DEFINE_integer(
    'replica_id',
    0,
    'Replica ID of the current process. This is used to distribute the '
    'evaluation across multiple machines.',
)
_TASK_NAME = flags.DEFINE_string(
    'task_name',
    'evaluation',
    'Name of the task to perform. This is used to name the dataset.',
)
_NUM_MACHINES = flags.DEFINE_integer(
    'num_machines',
    1,
    'Number of machines used to generate the dataset. This is used to '
    'distribute the dataset generation across multiple machines.',
)
_SEED = flags.DEFINE_integer('seed', None, 'Seed to use.')
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


def delete_dataset(dataset_path: str, dataset_id: str):
  """Deletes a dataset from the given path.

  Args:
      dataset_path: The path to the dataset.
  """
  dataset_path = os.path.join(dataset_path, dataset_id)
  shutil.rmtree(dataset_path)


def load_dataset(dataset_path: str, dataset_id: str):
  """Loads a dataset from the given path.

  Args:
      dataset_path: The path to the dataset.
      dataset_id: The name of the dataset.

  Returns:
      The loaded dataset.
  """
  os.environ['MINARI_DATASETS_PATH'] = dataset_path
  return minari.load_dataset(dataset_id)


def collect_dataset(
    env_name,
    checkpoint_path,
    dataset_path,
    dataset_id,
    num_episodes,
    seed,
):
  try:
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
      env_kwargs = {
          'mode': 'test',
      }
    else:
      data_collector_call = DataCollector
      env_kwargs = {}

    env = suite_gym.create_domain(
        env_name=env_name,
        root_dir=_ROOT_DIR.value,
        gym_env_wrappers=[data_collector_call],
        **env_kwargs,
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

    dataset = env.gym.create_dataset(
        dataset_id=dataset_id,
    )
    env.close()
    return dataset
  except Exception as e:
    import traceback

    logging.error('Error occurred: %s', e)
    logging.error(traceback.format_exc())
    return None


def main(_):
  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)

  job_completion_index = _REPLICA_ID.value
  logging.info('Job completion index: %s', job_completion_index)

  if _DATASETS_PATH.value is not None:
    base_path = _DATASETS_PATH.value
  else:
    base_path = _ROOT_DIR.value

  minari_datasets_path = os.path.join(
      os.path.expanduser(base_path),
      _TASK_NAME.value,
      _SKILL_LEVEL.value,
      f'{job_completion_index:03d}',
  )

  tmp_minari_datasets_path = os.path.join(
      '/tmp',
      _TASK_NAME.value,
      _SKILL_LEVEL.value,
      f'{job_completion_index:03d}',
  )
  root_dir = os.path.expanduser(_ROOT_DIR.value)

  os.environ['MINARI_DATASETS_PATH'] = tmp_minari_datasets_path
  logging.info('Set MINARI_DATASETS_PATH to %s', minari_datasets_path)

  evaluation_data_path = os.path.join(
      root_dir,
      _TASK_NAME.value,
      _SKILL_LEVEL.value,
      'evaluation_data_with_skill_levels.csv',
  )

  while True:
    if (
        os.path.exists(evaluation_data_path)
        and os.path.getsize(evaluation_data_path) > 0
    ):
      try:
        evaluation_df = pd.read_csv(evaluation_data_path)
        break
      except pd.errors.EmptyDataError:
        logging.warning('Evaluation File is empty, waiting to retry...')
    else:
      logging.info(
          'Waiting for evaluation data to be available at %s',
          evaluation_data_path,
      )

    time.sleep(10)

  evaluation_df = evaluation_df[evaluation_df.skill_level == _SKILL_LEVEL.value]
  if evaluation_df.empty:
    logging.warning('No policies found for skill level: %s', _SKILL_LEVEL.value)
    return

  logging.info(
      'After filtering by skill level, %s policies found.', len(evaluation_df)
  )

  num_episodes_to_generate = _NUM_EPISODES.value // _NUM_MACHINES.value
  remainder = _NUM_EPISODES.value % _NUM_MACHINES.value
  if job_completion_index < remainder:
    num_episodes_to_generate += 1

  evaluation_df = evaluation_df.sample(
      random_state=_SEED.value, n=_NUM_PROCESSES.value, replace=True
  )

  if num_episodes_to_generate == 0 or evaluation_df.empty:
    logging.warning('No episodes to generate.')
    return

  episodes_per_checkpoint = num_episodes_to_generate // len(evaluation_df)
  logging.info('Episodes per checkpoint: %s', episodes_per_checkpoint)
  remainder = num_episodes_to_generate % len(evaluation_df)
  logging.info(
      'After distributing episodes, %s episodes to generate.',
      num_episodes_to_generate,
  )
  num_episodes_list = [
      episodes_per_checkpoint + 1 if i < remainder else episodes_per_checkpoint
      for i in range(len(evaluation_df))
  ]

  dataset_paths = []
  dataset_ids = []
  env_name = _ENV_NAME.value[:-3]
  for i in range(_NUM_PROCESSES.value):
    tmp_dataset_id = f'{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-{job_completion_index:03d}-{i:03d}-v0'
    dataset_paths.append(os.path.join(tmp_minari_datasets_path, tmp_dataset_id))
    dataset_ids.append(tmp_dataset_id)

  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    tasks = zip(
        itertools.cycle([_ENV_NAME.value]),
        evaluation_df.checkpoint_path,
        dataset_paths,
        dataset_ids,
        num_episodes_list,
        itertools.cycle([_SEED.value]),
    )
    datasets = pool.starmap(collect_dataset, tasks)
    pool.close()
    pool.join()
  logging.info('Finished collecting all episodes.')

  logging.info('Combining local datasets...')
  dataset_id = f'{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-{job_completion_index:03d}-v0'
  dataset = minari.combine_datasets(
      datasets_to_combine=datasets, new_dataset_id=dataset_id
  )

  logging.info('Successfully combined datasets')
  logging.info('\tTotal steps: %s', dataset.total_steps)
  logging.info('\tTotal episodes: %s', dataset.total_episodes)

  logging.info('Moving dataset to final location %s', minari_datasets_path)
  os.makedirs(minari_datasets_path, exist_ok=True)
  subprocess.run(
      [
          'cp',
          '-r',
          os.path.join(tmp_minari_datasets_path, dataset_id),
          os.path.join(minari_datasets_path, dataset_id),
      ],
      check=True,
  )
  logging.info('Successfully moved dataset to %s ', minari_datasets_path)

  logging.info('Cleaning up temporary datasets.')
  with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
    tasks = zip(dataset_paths, dataset_ids)
    pool.starmap(delete_dataset, tasks)
    pool.close()
    pool.join()
  logging.info('Finished cleaning up temporary datasets.')

  if job_completion_index == 0:
    replica_dataset_paths = []
    replica_dataset_ids = []
    for i in range(_NUM_MACHINES.value):
      replica_dataset_id = (
          f'{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-{i:03d}-v0'
      )
      replica_dataset_paths.append(
          os.path.join(
              os.path.expanduser(base_path),
              _TASK_NAME.value,
              _SKILL_LEVEL.value,
              f'{i:03d}',
          )
      )
      replica_dataset_ids.append(replica_dataset_id)

    for path in replica_dataset_paths:
      while not os.path.exists(path):
        logging.info('Waiting for dataset %s to be available.', path)
        time.sleep(10)
    logging.info('All datasets are now available for combination.')

    # To use the replica datasets, we'll have to copy them to our local machine
    local_replica_dataset_paths = [
        os.path.join(
            '/tmp', _TASK_NAME.value, _SKILL_LEVEL.value, 'replicas', f'{i:03d}'
        )
        for i in range(_NUM_MACHINES.value)
    ]
    logging.info('Copying replica datasets to local machine...')
    for i, (replica_dataset_path, local_replica_dataset_path) in enumerate(
        zip(replica_dataset_paths, local_replica_dataset_paths)
    ):
      os.makedirs(local_replica_dataset_path, exist_ok=True)
      subprocess.run(
          [
              'cp',
              '-r',
              os.path.join(replica_dataset_path, replica_dataset_ids[i]),
              os.path.join(local_replica_dataset_path, replica_dataset_ids[i]),
          ],
          check=True,
      )
      logging.info('Finished copying replica dataset %s.', i)

    # Multiprocessing won't work with h5 operations, so we need to sequentially
    # load
    with multiprocessing.Pool(processes=1) as pool:
      replica_datasets = pool.starmap(
          load_dataset, zip(local_replica_dataset_paths, replica_dataset_ids)
      )
      pool.close()
      pool.join()

    logging.info('Finished loading all datasets.')

    logging.info('Combining datasets...')
    os.environ['MINARI_DATASETS_PATH'] = os.path.join(
        os.path.expanduser(base_path),
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
    )
    full_dataset = minari.combine_datasets(
        datasets_to_combine=replica_datasets,
        new_dataset_id=f'{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-v0',
    )
    logging.info('Successfully combined datasets from each replica.')
    logging.info('\tTotal steps: %s', full_dataset.total_steps)
    logging.info('\tTotal episodes: %s', full_dataset.total_episodes)

    logging.info('Cleaning up temporary datasets.')
    with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
      tasks = zip(replica_dataset_paths, replica_dataset_ids)
      pool.starmap(delete_dataset, tasks)
      pool.close()
      pool.join()


if __name__ == '__main__':
  app.run(main)
