from typing import Any
from typing import Union
from typing import OrderedDict
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.metrics import py_metrics
from tf_agents.policies import TFPolicy
from tf_agents.policies import policy_loader
from tf_agents.policies import random_py_policy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.tf_py_policy import TFPyPolicy
from tf_agents.train import actor
import tensorflow as tf
from tf_agents.trajectories import time_step as ts


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
  episode_reward_metric = py_metrics.AverageReturnMetric()
  rollout_actor = actor.Actor(
      env=env,
      train_step=policy._train_step_from_last_restored_checkpoint_path,
      policy=policy,
      observers=[episode_reward_metric],
      episodes_per_run=1,
  )

  episode_returns = []
  for _ in range(num_episodes):
    rollout_actor.run()
    episode_returns.append(episode_reward_metric.result())
    episode_reward_metric.reset()
  return np.array(episode_returns)


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


def apply_action_mask(policy: TFPolicy, mask_fn: callable, ) -> TFPolicy:
  pass


def create_random_py_policy(env: py_environment.PyEnvironment,
    observation_and_action_constraint_splitter: callable = None) -> RandomPyPolicy:
  """Creates a random policy for the given environment.

  Args:
      env: The environment for which the policy is to be created.

  Returns:
      A random policy for the given environment.
  """
  time_step_spec = env.time_step_spec()
  action_spec = env.action_spec()

  return random_py_policy.RandomPyPolicy(time_step_spec, action_spec,
                                         observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)


def create_tf_policy(policy: TFPolicy,
    observation_and_action_constraint_splitter: callable = None) -> TFPyPolicy:
  """Wraps an existing TFPolicy as a TFPyPolicy with an optional observation and action constraint splitter."""

  new_policy = TFPyPolicy(policy=policy,

                          )

  return TFPyPolicy(policy,
                    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)


def mask_circuit_training_actions(circuit_env, observation):
  mask = circuit_env.unwrapped._get_mask()
  return observation, mask


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
