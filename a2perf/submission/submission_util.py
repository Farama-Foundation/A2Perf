import enum
import functools
import importlib
import json
import multiprocessing
import os
import sys
import timeit
import typing
from contextlib import contextmanager

import gin
import numpy as np
from absl import flags
from absl import logging
from tf_agents.metrics import py_metrics
from tf_agents.train import actor

from a2perf.domains.tfa import suite_gym
from a2perf.metrics.system import codecarbon


@contextmanager
def working_directory(path):
  """Context manager for temporarily changing the working directory."""
  prev_cwd = os.getcwd()
  os.chdir(path)
  sys.path.insert(0, path)
  try:
    yield
  finally:
    os.chdir(prev_cwd)
    sys.path.remove(path)


def setup_subprocess_env(gin_config_str, absl_flags):
  # Parse the gin config
  gin.parse_config(gin_config_str)
  logging.info('Gin config parsed')

  # Register absl flags from the dictionary
  for flag_name, flag_value in absl_flags.items():
    if flag_name in flags.FLAGS:
      flags.FLAGS[flag_name].value = flag_value
      logging.info('Flag %s set to %s', flag_name, flag_value)
    else:
      logging.warning('Flag %s not found', flag_name)


def _load_spec(module_path, filename):
  """Loads the spec from the given module path."""
  participant_file_path = os.path.join(module_path, filename)
  spec = importlib.util.spec_from_file_location(
      f'{filename}', participant_file_path
  )
  return spec


def _load_module(module_path, filename):
  """Loads the module from the given module path."""
  spec = _load_spec(module_path, filename)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module, spec


def _load_policy(module_path, env):
  """Loads the policy from the participant's module."""
  with working_directory(module_path):
    participant_module, participant_module_spec = _load_module(
        module_path, 'inference.py')
    policy = participant_module.load_policy(env)
  return policy, participant_module


def perform_rollouts(
    module_path,
    create_domain_fn,
    num_episodes=1,
    gin_config_str=None,
    absl_flags=None,
    rollout_rewards_queue=None
):
  """Performs rollouts using the given policy.
  
  Args:
      create_domain_fn: Function that creates the domain.
      preprocess_obs_fn: Function that preprocesses the observation.
      infer_once_fn: Function that performs inference.
      num_episodes: Number of episodes to perform rollouts.
      policy: Policy to use for performing rollouts.
      gin_config_str: Gin config string to use for creating the domain.
        
  Returns:
      List of rewards from each episode.
  """
  setup_subprocess_env(gin_config_str, absl_flags)
  env = create_domain_fn()
  policy, participant_module = _load_policy(module_path, env)
  episode_reward_metric = py_metrics.AverageReturnMetric()
  rollout_actor = actor.Actor(
      env=env,
      train_step=policy._train_step_from_last_restored_checkpoint_path,
      policy=policy,
      observers=[episode_reward_metric],
      episodes_per_run=1, )

  all_rewards = []
  for _ in range(num_episodes):
    rollout_actor.run()
    all_rewards.append(episode_reward_metric.result())
    episode_reward_metric.reset()

  if rollout_rewards_queue:
    for reward in all_rewards:
      rollout_rewards_queue.put(reward)

  return all_rewards


def train(
    module_path,
    gin_config_str=None,
    absl_flags=None
):
  """Trains the participant's policy."""
  setup_subprocess_env(gin_config_str, absl_flags)
  with working_directory(module_path):
    participant_module, participant_module_spec = _load_module(
        module_path, 'train.py')
    participant_module.train()


@gin.constants_from_enum
class BenchmarkMode(enum.Enum):
  TRAIN = 'train'
  INFERENCE = 'inference'


@gin.constants_from_enum
class BenchmarkDomain(enum.Enum):
  WEB_NAVIGATION = 'WebNavigation-v0'
  CIRCUIT_TRAINING = 'CircuitTraining-v0'
  QUADRUPED_LOCOMOTION = 'QuadrupedLocomotion-v0'


@gin.constants_from_enum
class SystemMetrics(enum.Enum):
  INFERENCE_TIME = 'InferenceTime'
  TRAINING_TIME = 'TrainingTime'
  MEMORY_USAGE = 'MemoryUsage'


@gin.constants_from_enum
class ReliabilityMetrics(enum.Enum):
  IqrWithinRuns = 'IqrWithinRuns'
  IqrAcrossRuns = 'IqrAcrossRuns'
  LowerCVaROnDiffs = 'LowerCVaROnDiffs'
  LowerCVaROnDrawdown = 'LowerCVaROnDrawdown'
  LowerCVarOnAcross = 'LowerCVarOnAcross'
  MedianPerfDuringTraining = 'MedianPerfDuringTraining'
  MadAcrossRollouts = 'MadAcrossRollouts'
  IqrAcrossRollouts = 'IqrAcrossRollouts'
  StddevAcrossRollouts = 'StddevAcrossRollouts'
  UpperCVaRAcrossRollouts = 'UpperCVaRAcrossRollouts'
  LowerCVaRAcrossRollouts = 'LowerCVaRAcrossRollouts'


@gin.configurable
class Submission:

  def __init__(
      self,
      root_dir: str,
      metric_values_dir: str,
      participant_module_path: str,
      mode: BenchmarkMode,
      domain: BenchmarkDomain,
      train_logs_dirs: typing.List[str] = None,
      num_inference_steps: int = 1000,
      num_inference_episodes: int = 1,
      time_participant_code: bool = True,
      measure_emissions: bool = False,
      baseline_measure_sec: float = 0,
      plot_metrics: bool = True,
      run_offline_metrics_only: bool = False,
      reliability_metrics: typing.List[ReliabilityMetrics] = None,
      tracking_mode: str = None,
  ):
    """Object that represents a submission to the benchmark.

    Args:
        participant_module_path: Path to the module that contains the
          participant's code.
        mode: Benchmark mode (train or inference).
        domain: Benchmark domain (web navigation, circuit training or quadruped
          locomotion).
        root_dir: Root directory for the submission.
        metric_values_dir: Directory where the metric values will be saved.
        train_logs_dirs: List of directories where the training logs are saved.
          Relative to the root directory.
        num_inference_steps: Number of steps to run inference for.
        num_inference_episodes: Number of episodes to run inference for.
        time_participant_code: Whether to time the participant's code.
        measure_emissions: Whether to measure emissions.
        baseline_measure_sec: Baseline time to measure emissions for.
        plot_metrics: Whether to plot the metrics.
        run_offline_metrics_only: Whether to run only the offline metrics.
        reliability_metrics: List of reliability metrics to compute.
        tracking_mode: Tracking mode for the participant's code.
    """
    self.root_dir = root_dir
    self.metric_values_dir = metric_values_dir
    self.train_logs_dirs = train_logs_dirs
    os.makedirs(self.root_dir, exist_ok=True)
    os.makedirs(self.metric_values_dir, exist_ok=True)
    if self.train_logs_dirs is not None:
      self.train_logs_dirs = [
          os.path.join(self.root_dir, train_logs_dir)
          for train_logs_dir in self.train_logs_dirs
      ]
    self.run_offline_metrics_only = run_offline_metrics_only
    self.baseline_measure_sec = baseline_measure_sec
    self.tracking_mode = tracking_mode
    self.mp_context = multiprocessing.get_context('spawn')
    self.gin_config_str = None
    self.absl_flags = None

    self.measure_emissions = measure_emissions
    self.plot_metrics = plot_metrics
    self.num_inference_steps = num_inference_steps
    self.num_inference_episodes = num_inference_episodes
    self.time_inference_steps = time_participant_code
    self.participant_module_path = os.path.abspath(participant_module_path)
    self.domain = domain
    self.mode = mode
    self.reliability_metrics = reliability_metrics

    self.metrics_results = {}
    metrics_path = os.path.join(
        self.metric_values_dir,
        'inference_metrics.json'
        if self.mode == BenchmarkMode.INFERENCE
        else 'train_metrics.json',
    )
    if os.path.exists(metrics_path):
      logging.info(f'Loading pre-existing metric results from {metrics_path}')
      with open(metrics_path, 'r') as f:
        self.metrics_results = json.load(f)

  def _get_observation_data(self, env):
    data = []
    for _ in range(self.num_inference_steps):
      observation = env.observation_space.sample()
      data.append(observation)
    return data

  def _train(
      self,
  ):
    setup_subprocess_env(self.gin_config_str, self.absl_flags)

    @codecarbon.track_emissions(output_dir=self.metric_values_dir,
                                output_file='train_emissions.csv')
    def train_and_track_emissions():
      train_process = self.mp_context.Process(
          target=train,
          args=(
              self.participant_module_path, self.gin_config_str,
              self.absl_flags)
      )
      train_process.start()
      train_process.join()

    if self.measure_emissions:
      return train_and_track_emissions()
    else:
      return train(self.participant_module_path, self.gin_config_str,
                   self.absl_flags)

  def _perform_rollouts(self,
      num_episodes,
      measure_emissions,
      output_dir,
      rollout_rewards_queue
  ):
    """
    Perform rollouts and optionally track emissions.

    Args:
        num_episodes: Number of episodes to perform rollouts.
        measure_emissions: Flag to indicate if emissions should be measured.
        output_dir: Directory to save the emissions data.

    Returns:
        List of rewards from each episode.
    """
    setup_subprocess_env(self.gin_config_str, self.absl_flags)

    create_domain_fn = functools.partial(suite_gym.create_domain,
                                         env_name=self.domain.value,
                                         root_dir=self.root_dir)
    if measure_emissions:
      @codecarbon.track_emissions(output_dir=output_dir,
                                  output_file='inference_emissions.csv')
      def perform_rollouts_and_track_emissions():
        rollout_process = multiprocessing.Process(
            target=perform_rollouts,
            args=(
                self.participant_module_path,
                create_domain_fn,
                num_episodes,
                self.gin_config_str,
                self.absl_flags,
                rollout_rewards_queue
            )
        )
        rollout_process.start()
        rollout_process.join()

      return perform_rollouts_and_track_emissions()
    else:
      return perform_rollouts(create_domain_fn=create_domain_fn,
                              num_episodes=num_episodes,
                              module_path=self.participant_module_path,
                              gin_config_str=self.gin_config_str,
                              absl_flags=self.absl_flags)

  def _run_training_benchmark(self):
    if not self.run_offline_metrics_only:
      participant_training_process = self.mp_context.Process(
          target=self._train,
      )

      participant_training_process.start()
      logging.info(
          f'Participant training process ID: {participant_training_process.pid}')

      participant_training_process.join()
      logging.info(
          f'Participant module process {participant_training_process.pid} finished'
      )

      if participant_training_process.is_alive():
        logging.error('Participant process is still running')
      elif participant_training_process.exitcode != 0:
        logging.error(
            'Participant process exited with code'
            f' {participant_training_process.exitcode}'
        )
      else:
        logging.info(
            f'Participant process {participant_training_process.pid} finished')

  def _run_inference_benchmark(self):
    if not self.run_offline_metrics_only:
      logging.info('Creating Gymnasium domain...')
      env = suite_gym.create_domain(env_name=self.domain.value,
                                    root_dir=self.root_dir)
      logging.info('Successfully created domain')


      logging.info('Generating inference data...')
      inference_data = self._get_observation_data(env)
      logging.info('Successfully generated inference data')

      metric_results = {}

      logging.info('Loading the policy for inference...')
      participant_policy, participant_module = _load_policy(
          module_path=self.participant_module_path,
          env=env)

      # Only include time_step_spec if the participant policy has it as an
      # attribute. This will be useful for participants using TF agents.
      time_step_spec = getattr(participant_policy, 'time_step_spec', None)
      preprocessed_data = [
          participant_module.preprocess_observation(x,
                                                    time_step_spec=time_step_spec)
          for x
          in inference_data]
      logging.info('Finished preprocessing the observation data')

      if self.time_inference_steps:
        logging.info('Timing inference steps...')
        inference_times = []
        for i in range(self.num_inference_steps):
          inference_step = lambda: participant_module.infer_once(
              policy=participant_policy,
              preprocessed_observation=preprocessed_data[i]
          )
          inference_times.append(timeit.timeit(inference_step, number=1))
        logging.info('Finished timing inference steps')

        metric_results['inference_time'] = {
            'values': inference_times,
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'max': np.max(inference_times),
            'median': np.median(inference_times),
            'min': np.min(inference_times)
        }

      # Running rollouts in a subprocess
      rollout_returns_queue = multiprocessing.Queue()
      rollout_process = multiprocessing.Process(
          target=self._perform_rollouts,
          args=(self.num_inference_episodes,
                self.measure_emissions,
                self.metric_values_dir,
                rollout_returns_queue)
      )

      rollout_process.start()
      rollout_process.join()

      all_rewards = []
      while not rollout_returns_queue.empty():
        all_rewards.append(rollout_returns_queue.get())

      print(f'All rewards: {all_rewards}')
      metric_results['rollout_returns'] = {
          'values': [float(reward) for reward in all_rewards],
          'mean': np.mean(all_rewards).astype(float),
          'std': np.std(all_rewards).astype(float),
          'max': np.max(all_rewards).astype(float),
          'median': np.median(all_rewards).astype(float),
          'min': np.min(all_rewards).astype(float),
      }

      logging.info('Metrics Results: %s', metric_results)
      with open(os.path.join(self.metric_values_dir,
                             'inference_metrics_results.json'), 'w') as f:
        json.dump(metric_results, f)

  def run_benchmark(self):
    # Gin configs and absl flags must be saved to pass to subprocesses
    self.gin_config_str = gin.config_str()
    self.absl_flags = {name: flags.FLAGS[name].value for name in flags.FLAGS}

    if not os.path.exists(self.participant_module_path):
      raise FileNotFoundError(
          f'Participant module path {self.participant_module_path} not found. This is necessary for running training and inference code.'
      )

    if self.mode == BenchmarkMode.TRAIN:
      self._run_training_benchmark()
    elif self.mode == BenchmarkMode.INFERENCE:

      if not os.path.exists(self.root_dir):
        raise FileNotFoundError(
            f'Root directory {self.root_dir} not found. This is necessary for loading the trained model'
        )
      self._run_inference_benchmark()
    else:
      raise ValueError('Benchmark mode must be either train or inference')
