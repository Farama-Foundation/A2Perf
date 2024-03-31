import functools
import json
import multiprocessing
import os
from typing import Any
from typing import Dict
from typing import OrderedDict
from typing import Union

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.environments import py_environment
from tf_agents.policies import policy_loader
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import time_step as ts

# noinspection PyUnresolvedReferences
from a2perf.domains import circuit_training
# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion
# noinspection PyUnresolvedReferences
from a2perf.domains import web_navigation
from a2perf.domains.utils.suite_gym import create_domain

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


def perform_rollouts(
    policy: TFPolicy, env: py_environment.PyEnvironment, num_episodes: int
) -> np.ndarray:
  """Perform rollouts using the policy and environment.

  Args:
      policy: The policy to use.
      env: The environment to use.
      num_episodes: The number of episodes to perform.

  Returns:
      The returns for each episode.
  """

  obs = env.reset()
  episode_returns = []
  for _ in range(num_episodes):
    episode_return = 0
    while not obs.is_last():
      obs = preprocess_observation(
          observation=obs.observation, time_step_spec=policy.time_step_spec
      )
      action = policy.action(obs)
      obs = env.step(action.action)
      episode_return += obs.reward
    episode_returns.append(episode_return)

  return np.array(episode_returns)


def load_policy_and_perform_rollouts(
    checkpoint_path: str, env_name: str, policy_path: str, num_episodes: int
) -> Dict[str, Any]:
  policy = load_policy(policy_path, checkpoint_path)
  env = create_domain(env_name)
  episode_returns = perform_rollouts(policy, env, num_episodes)

  eval_dict = {
      checkpoint_path: {
          'mean': np.mean(episode_returns),
          'std': np.std(episode_returns),
          'min': np.min(episode_returns),
          'max': np.max(episode_returns),
          'median': np.median(episode_returns),
          'count': episode_returns.size,
      }
  }
  logging.info(f'Evaluation for {checkpoint_path}: {eval_dict}')
  return eval_dict


def main(_):
  saved_model_path = os.path.join(_ROOT_DIR.value, 'policies',
                                  _POLICY_NAME.value)
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

  # Save as JSON
  evaluation_save_path = os.path.join(
      _ROOT_DIR.value, 'policies', 'evaluation.json'
  )
  with open(evaluation_save_path, 'w') as f:
    json.dump(all_episode_returns, f, indent=2)


if __name__ == '__main__':
  app.run(main)
