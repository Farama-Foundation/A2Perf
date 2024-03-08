import functools
import json
import multiprocessing
import os
from typing import Any
from typing import Dict

import numpy as np
from absl import app
from absl import flags
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.policies import policy_loader
from tf_agents.policies.tf_policy import TFPolicy

# noinspection PyUnresolvedReferences
from a2perf.domains import circuit_training
# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion
# noinspection PyUnresolvedReferences
from a2perf.domains import web_navigation

_NUM_EVAL_EPISODES = flags.DEFINE_integer(
    'num_eval_episodes', 100, 'Number of episodes to evaluate the policy.'
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


def create_domain(env_name) -> py_environment.PyEnvironment:
  if env_name == 'CircuitTraining-v0':

    netlist_path = os.environ.get('NETLIST_PATH', None)
    init_placement_path = os.environ.get('INIT_PLACEMENT_PATH', None)
    kwargs = {
        'netlist_path': netlist_path,
        'init_placement_path': init_placement_path,
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

  return suite_gym.load(env_name, gym_kwargs=kwargs)


def load_policy(saved_model_path: str, checkpoint_path: str) -> TFPolicy:
  """Loads a policy model from the environment's root directory.

  Args:
      saved_model_path: The path of a directory containing a full saved model.
      checkpoint_path: The path to a directory that contains variable checkpoints
        (as opposed to full saved models) for the policy.

  Returns:
      The loaded policy.
  """
  policy = policy_loader.load(
      saved_model_path=saved_model_path,
      checkpoint_path=checkpoint_path,
  )
  return policy


def load_policy_and_perform_rollouts(checkpoint_path: str, env_name: str,
    policy_path: str,
    num_episodes: int) -> Dict[str, Any]:
  policy = load_policy(policy_path, checkpoint_path)
  env = create_domain(env_name)

  obs = env.reset()
  episode_returns = []
  for _ in range(num_episodes):
    episode_return = 0
    while not obs.is_last():
      action = policy.action(obs)
      obs = env.step(action.action)
      episode_return += obs.reward
    episode_returns.append(episode_return)

  episode_returns = np.array(episode_returns)

  return {
      checkpoint_path: {
          'mean': np.mean(episode_returns),
          'std': np.std(episode_returns),
          'min': np.min(episode_returns),
          'max': np.max(episode_returns),
          'median': np.median(episode_returns),
      }
  }


def main(_):
  saved_model_path = os.path.join(_ROOT_DIR.value, 'policies', 'policy')
  checkpoints_path = os.path.join(_ROOT_DIR.value, 'policies', 'checkpoints')

  # Get absolute paths of all checkpoints
  all_checkpoints_paths = [
      os.path.join(checkpoints_path, checkpoint)
      for checkpoint in os.listdir(checkpoints_path)
  ]

  # Create a partial function that has all the fixed parameters set
  partial_func = functools.partial(
      load_policy_and_perform_rollouts,
      env_name=_ENV_NAME.value, policy_path=saved_model_path,
      num_episodes=_NUM_EVAL_EPISODES.value
  )

  with multiprocessing.Pool() as pool:
    episode_returns = pool.map(
        partial_func, all_checkpoints_paths
    )

  all_episode_returns = {k: v for d in episode_returns for k, v in d.items()}

  # Save as JSON
  evaluation_save_path = os.path.join(_ROOT_DIR.value, 'policies',
                                      'evaluation.json')
  with open(evaluation_save_path, 'w') as f:
    json.dump(all_episode_returns, f, indent=2)


if __name__ == '__main__':
  app.run(main)
