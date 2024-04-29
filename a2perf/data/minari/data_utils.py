import functools
import os
import shutil

from a2perf.domains.tfa import suite_gym
from a2perf.domains.tfa.utils import load_policy
from a2perf.domains.tfa.utils import perform_rollouts
import minari
import numpy as np
import tensorflow as tf
import gymnasium as gym
from minari import DataCollector
from absl import logging


def collect_dataset(
    env_name,
    root_dir,
    checkpoint_path,
    dataset_path,
    dataset_id,
    num_episodes,
    seed,
    policy_name='policy',
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
        root_dir=root_dir,
        gym_env_wrappers=[data_collector_call],
        **env_kwargs,
    )

    # checkpoint_path = checkpoint_path.replace('/gcs', os.path.expanduser('~/gcs'))
    saved_model_path = os.path.join(
        checkpoint_path, '..', '..', policy_name
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


def combine_minari_datasets(dataset_a, dataset_b, dataset_id):
  """Combines two minari datasets.

  Args:
      dataset_a: The first dataset to combine.
      dataset_b: The second dataset to combine.
      dataset_id: The ID of the combined dataset.

  Returns:
      The combined dataset.
  """
  combined_dataset = minari.combine_datasets(
      datasets_to_combine=[dataset_a, dataset_b],
      new_dataset_id=dataset_id,
  )
  return combined_dataset
