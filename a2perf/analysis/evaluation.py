import functools
import json
import multiprocessing
import os
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
from absl import app
from absl import flags
from absl import logging

from a2perf.analysis.metrics_lib import load_training_system_data
# noinspection PyUnresolvedReferences
from a2perf.domains import circuit_training
# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion
# noinspection PyUnresolvedReferences
from a2perf.domains import web_navigation
from a2perf.domains.tfa.suite_gym import create_domain
from a2perf.domains.tfa.utils import load_policy
from a2perf.domains.tfa.utils import perform_rollouts

_NUM_EVAL_EPISODES = flags.DEFINE_integer(
    'num_eval_episodes', 100, 'Number of episodes to evaluate the policy.'
)

_MAX_PARALLEL_ENVS = flags.DEFINE_integer(
    'max_parallel_envs', 1, 'Maximum number of parallel environments to use.'
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory of the environment. If not set, the ROOT_DIR environment '
    'variable is used.',
)

_ENV_NAME = flags.DEFINE_string(
    'env_name', 'CartPole-v0', 'The name of the environment to evaluate.'
)
_POLICY_NAME = flags.DEFINE_string(
    'policy_name', 'policy', 'The name of the policy to evaluate.'
)


def load_policy_and_perform_rollouts(
    checkpoint_path: str, env_name: str, policy_path: str, num_episodes: int
) -> Dict[str, Any]:
  policy = load_policy(policy_path, checkpoint_path)
  env = create_domain(env_name)
  episode_returns = perform_rollouts(policy, env, num_episodes)

  eval_dict = {
      checkpoint_path: {
          'mean': np.mean(episode_returns).astype(float),
          'std': np.std(episode_returns).astype(float),
          'min': np.min(episode_returns).astype(float),
          'max': np.max(episode_returns).astype(float),
          'median': np.median(episode_returns).astype(float),
          'count': int(episode_returns.size),
          'rollout_returns': [float(v) for v in episode_returns],
      }
  }

  logging.info('Evaluation results for %s:', checkpoint_path)
  logging.info('\t%s', eval_dict[checkpoint_path])
  return eval_dict


def add_training_energy_cost(checkpoint_item: Tuple[str, Dict[str, Any]],
    total_energy_kwh) -> Tuple[str, Dict[str, Any]]:
  checkpoint_path = checkpoint_item[0]
  checkpoint_dict = checkpoint_item[1]

  policy_checkpoint_name = os.path.basename(checkpoint_path)
  policy_checkpoint_number = int(policy_checkpoint_name.split('_')[-1])

  checkpoint_dict.update(
      {'total_training_energy_kwh': total_energy_kwh,
       'training_energy_kwh': total_energy_kwh * policy_checkpoint_number,
       'checkpoint_number': policy_checkpoint_number})
  return checkpoint_path, checkpoint_dict


def main(_):
  multiprocessing.set_start_method('spawn', force=False)
  saved_model_path = os.path.join(
      _ROOT_DIR.value, 'policies', _POLICY_NAME.value
  )
  checkpoints_path = os.path.join(_ROOT_DIR.value, 'policies', 'checkpoints')

  # Get absolute paths of all checkpoints
  all_checkpoints_paths = [
      os.path.join(checkpoints_path, checkpoint)
      for checkpoint in os.listdir(checkpoints_path)
  ]

  # Create a partial function that has all the fixed parameters set
  partial_func = functools.partial(
      load_policy_and_perform_rollouts,
      env_name=_ENV_NAME.value,
      policy_path=saved_model_path,
      num_episodes=_NUM_EVAL_EPISODES.value,
  )

  with multiprocessing.Pool(_MAX_PARALLEL_ENVS.value) as pool:
    episode_returns = pool.map(partial_func, all_checkpoints_paths)
    pool.close()
    pool.join()

  all_episode_returns = {k: v for d in episode_returns for k, v in d.items()}

  # Use the experiment path to get the system metrics for this training run
  experiment_path = os.path.abspath(
      os.path.join(_ROOT_DIR.value, os.pardir, os.pardir, os.pardir, os.pardir))
  logging.debug('Experiment path: %s', experiment_path)
  training_system_df = load_training_system_data(
      base_dir=os.path.abspath(os.path.join(experiment_path, os.pardir)),
      experiment_ids=[os.path.basename(experiment_path)])

  # For each run-id, take the last `energy_consumed` entry and sum them together
  total_training_energy_kwh = training_system_df.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']
  )['energy_consumed'].last().sum()

  # Add the training sample cost to the evaluation results
  with multiprocessing.Pool(_MAX_PARALLEL_ENVS.value) as pool:
    all_episode_returns = pool.map(
        functools.partial(add_training_energy_cost,
                          total_energy_kwh=total_training_energy_kwh),
        all_episode_returns.items()
    )
    pool.close()
    pool.join()

  # Turn all_episode_returns back into a dictionary
  all_episode_returns = {k: v for (k, v) in all_episode_returns}

  maximum_checkpoint_number = max(
      [v['checkpoint_number'] for v in all_episode_returns.values()]
  )
  logging.info('Maximum checkpoint number: %d', maximum_checkpoint_number)

  for checkpoint_path, checkpoint_dict in all_episode_returns.items():
    # Adjusting the training energy cost such that earlier checkpoints are
    # associated with less energy usage
    checkpoint_dict['training_energy_kwh'] = checkpoint_dict[
                                               'training_energy_kwh'] / maximum_checkpoint_number

    # Make sure that the energy usage for the final checkpoint is the same as
    # the total energy usage
    if checkpoint_dict['checkpoint_number'] == maximum_checkpoint_number:
      assert checkpoint_dict['training_energy_kwh'] == total_training_energy_kwh

  # Save as JSON
  evaluation_save_path = os.path.join(
      _ROOT_DIR.value, 'policies', 'evaluation.json'
  )
  with open(evaluation_save_path, 'w') as f:
    json.dump(all_episode_returns, f, indent=2)


if __name__ == '__main__':
  app.run(main)
