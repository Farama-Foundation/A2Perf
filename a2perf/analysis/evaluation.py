import os
from typing import Any
from typing import List

from absl import app
from absl import flags
from absl import logging
from tf_agents.policies import policy_loader
from tf_agents.policies.tf_policy import TFPolicy
import multiprocessing

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


def perform_rollouts(policy: TFPolicy, num_episodes: int) -> List[Any]:
  """Runs rollouts using the given policy.

  Args:
      policy: The policy to use for rollouts.
      num_episodes: The number of episodes to run.

  Returns:
      The results of the rollouts.
  """
  rollouts_returns = []

  env = g

  results = []
  for _ in range(num_episodes):
    results.append(policy.rollout())
  return results


def evaluate_policy(saved_model_path: str, checkpoint_path: str) -> None:
  """Evaluates a policy and saves the results to a file.

  Args:
      saved_model_path: The path of a directory containing a full saved model.
      checkpoint_path: The path to a directory that contains variable checkpoints
        (as opposed to full saved models) for the policy.
  """
  # Load the policy
  policy = load_policy(saved_model_path, checkpoint_path)

  # Evaluate the policy


def main(_):
  saved_model_path = os.path.join(_ROOT_DIR.value, 'policies', 'policy')
  checkpoints_path = os.path.join(_ROOT_DIR.value, 'policies', 'checkpoints')

  # Get absolute paths of all checkpoints
  all_checkpoints_paths = [
      os.path.join(checkpoints_path, checkpoint)
      for checkpoint in os.listdir(checkpoints_path)
  ]


if __name__ == '__main__':
  app.run(main)
