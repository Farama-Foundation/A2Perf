import csv
import enum
import importlib
import json
import logging
import multiprocessing
import os
import sys
import timeit
import typing
from contextlib import contextmanager

import gin
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from rl_perf.metrics.reliability.rl_reliability_metrics.evaluation.eval_metrics import \
  Evaluator
from rl_perf.metrics.reliability.rl_reliability_metrics.metrics import (
  IqrAcrossRuns, IqrWithinRuns,
  LowerCVaROnAcross,
  LowerCVaROnDiffs,
  IqrAcrossRollouts, MedianPerfDuringTraining
, StddevAcrossRollouts, MadAcrossRollouts, UpperCVaRAcrossRollouts,
  LowerCVaRAcrossRollouts, )
from rl_perf.metrics.system import codecarbon
from rl_perf.metrics.system.profiler.base_profiler import BaseProfiler


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


def _start_profilers(profilers: typing.List[typing.Type[BaseProfiler]],
    profiler_events: typing.List[multiprocessing.Event],
    participant_event: multiprocessing.Event,
    participant_process: multiprocessing.Process, log_dir: str, mp_context) -> \
    typing.List[multiprocessing.Process]:
  processes = []
  for profiler_class, profiler_event in zip(profilers, profiler_events):
    logging.info(f'Starting profiler: {profiler_class}')
    profiler = profiler_class(participant_process_event=participant_event,
                              profiler_event=profiler_event,
                              participant_process=participant_process,
                              base_log_dir=log_dir)
    profiler_process = mp_context.Process(target=profiler.start)
    profiler_process.start()
    processes.append(profiler_process)
  return processes


def _start_inference_profilers(participant_event, profilers, pipes,
    profiler_started_events, base_log_dir, mp_context):
  processes = []
  profiler_objects = []
  for profiler_class, pipe, profiler_event in zip(profilers, pipes,
                                                  profiler_started_events):
    logging.info(f'Starting profiler: {profiler_class}')
    profiler = profiler_class(pipe_for_participant_process=pipe,
                              profiler_event=profiler_event,
                              participant_process_event=participant_event,
                              base_log_dir=base_log_dir)
    profiler_objects.append(profiler)
    profiler_process = mp_context.Process(target=profiler.start, )
    profiler_process.start()
    processes.append(profiler_process)
  return processes, profiler_objects


@gin.configurable
class Submission:
  def __init__(self,
      root_dir: str,
      participant_module_path: str = None,
      profilers: typing.List[typing.Type[BaseProfiler]] = None,
      mode: BenchmarkMode = BenchmarkMode.TRAIN,
      domain: BenchmarkDomain = BenchmarkDomain.WEB_NAVIGATION,
      metric_values_dir: str = None,
      train_logs_dirs: typing.List[str] = None,
      num_inference_steps: int = 1000,
      num_inference_episodes: int = 1,
      time_participant_code: bool = True,
      measure_emissions: bool = False,
      baseline_measure_sec: float = 0,
      plot_metrics: bool = True,
      run_offline_metrics_only: bool = False,
      reliability_metrics: typing.List[ReliabilityMetrics] = None,
      tracking_mode: str = None):
    """Object that represents a submission to the benchmark.

    Args:
        participant_module_path: Path to the module that contains the participant's code.
        profilers: List of profilers to use.
        mode: Benchmark mode (train or inference).
        domain: Benchmark domain (web navigation, circuit training or quadruped locomotion).
        root_dir: Root directory for the submission.
        metric_values_dir: Directory where the metric values will be saved.
        train_logs_dirs: List of directories where the training logs are saved. Relative to the root directory.
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
    if self.metric_values_dir is None:
      self.metric_values_dir = os.path.join(self.root_dir, 'metrics')
    os.makedirs(self.metric_values_dir, exist_ok=True)
    if self.train_logs_dirs is not None:
      self.train_logs_dirs = [os.path.join(self.root_dir, train_logs_dir) for
                              train_logs_dir in
                              self.train_logs_dirs]
    self.run_offline_metrics_only = run_offline_metrics_only
    self.baseline_measure_sec = baseline_measure_sec
    self.tracking_mode = tracking_mode
    self.mp_context = multiprocessing.get_context('spawn')
    self.gin_config_str = None

    self.measure_emissions = measure_emissions
    self.plot_metrics = plot_metrics
    self.num_inference_steps = num_inference_steps
    self.num_inference_episodes = num_inference_episodes
    self.time_inference_steps = time_participant_code
    self.profilers = profilers if profilers is not None else []
    for profiler in self.profilers:
      profiler.base_log_dir = self.root_dir
    self.participant_module_path = os.path.abspath(participant_module_path)
    self.domain = domain
    self.mode = mode
    self.reliability_metrics = reliability_metrics

    self.metrics_results = {}
    metrics_path = os.path.join(self.metric_values_dir,
                                'inference_metrics.json' if self.mode == BenchmarkMode.INFERENCE else 'train_metrics.json')
    if os.path.exists(metrics_path):
      logging.info(
          f'Loading pre-existing metric results from {metrics_path}')
      with open(metrics_path, 'r') as f:
        self.metrics_results = json.load(f)

  def _load_participant_spec(self, filename):
    """Loads the participant spec from the participant module path."""
    participant_file_path = os.path.join(self.participant_module_path, filename)
    spec = importlib.util.spec_from_file_location(f"{filename}",
                                                  participant_file_path)
    return spec

  def _load_participant_module(self, filename):
    """Load the participant module and return the module object."""
    spec = self._load_participant_spec(filename)
    participant_module = importlib.util.module_from_spec(spec)
    return participant_module, spec

  @gin.configurable("Submission.create_domain")
  def create_domain(self, **kwargs):
    if self.domain == BenchmarkDomain.WEB_NAVIGATION:
      from rl_perf.domains.web_nav.gwob.CoDE import vocabulary_node

      if kwargs.get('reload_vocab', False):
        global_vocab_dict = np.load(
            os.path.join(self.root_dir, 'train', 'global_vocab.npy'),
            allow_pickle=True).item()
        global_vocab = vocabulary_node.LockedVocabulary()
        global_vocab.restore(dict(global_vocab=global_vocab_dict))
        kwargs['global_vocabulary'] = global_vocab
        kwargs.pop('reload_vocab')
    elif self.domain == BenchmarkDomain.CIRCUIT_TRAINING:
      pass
    elif self.domain == BenchmarkDomain.QUADRUPED_LOCOMOTION:
      from rl_perf.domains import quadruped_locomotion
    else:
      raise NotImplementedError(f'Domain {self.domain} not implemented')
    return gym.make(id=self.domain.value, **kwargs)

  def _get_observation_data(self, env):
    data = []
    for _ in range(self.num_inference_steps):
      observation = env.observation_space.sample()
      data.append(observation)
    return data

  def _train(self, participant_event: multiprocessing.Event,
      profiler_events: typing.List[multiprocessing.Event]):
    gin.parse_config(self.gin_config_str)
    participant_event.set()

    # Wait for all profilers to start up before continuing
    for profiler_event in profiler_events:
      profiler_event.wait()

    with working_directory(self.participant_module_path):
      participant_module, participant_module_spec = self._load_participant_module(
          'train.py')
      print(self.participant_module_path)
      print(participant_module_spec)
      participant_module_spec.loader.exec_module(participant_module)

      if self.measure_emissions:
        @codecarbon.track_emissions(output_dir=self.metric_values_dir,
                                    output_file='train_emissions.csv', )
        def train():
          return participant_module.train()

        train()
      else:
        participant_module.train()

    participant_event.clear()
    for profiler_event in profiler_events:
      profiler_event.clear()

  def _inference(self, participant_event: multiprocessing.Event,
      profiler_events: typing.List[multiprocessing.Event],
      inference_data: typing.List[typing.Any],
      rollout_data_queue: multiprocessing.Queue, ):
    gin.parse_config(self.gin_config_str)
    participant_event.set()

    env = self.create_domain()
    metric_results = {}
    with working_directory(self.participant_module_path):
      participant_module, participant_module_spec = self._load_participant_module(
          'inference.py')
      print(self.participant_module_path)
      print(participant_module_spec)
      participant_module_spec.loader.exec_module(participant_module)
      participant_model = participant_module.load_model(env=env)

      preprocessed_data = [participant_module.preprocess_observation(x) for x in
                           inference_data]

      def inference_step():
        return participant_module.infer_once(model=participant_model,
                                             observation=preprocessed_data[i])

      if self.time_inference_steps:
        inference_times = []
        for i in range(self.num_inference_steps):
          inference_times.append(timeit.timeit(inference_step, number=1))

        metric_results['inference_time'] = dict(values=inference_times,
                                                mean=np.mean(
                                                    inference_times),
                                                std=np.std(
                                                    inference_times))

      def perform_rollouts():
        all_rewards = []
        for _ in range(self.num_inference_episodes):
          observation, info = env.reset()
          terminated = False
          truncated = False
          rewards = 0
          while not terminated and not truncated:
            preprocessed_obs = participant_module.preprocess_observation(
                observation)
            action = participant_module.infer_once(model=participant_model,
                                                   observation=preprocessed_obs)
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
          all_rewards.append(rewards)
          print(f'Episode reward: {rewards}')
        return all_rewards

      if self.measure_emissions:
        @codecarbon.track_emissions(output_dir=self.metric_values_dir,
                                    output_file='inference_emissions.csv', )
        def perform_rollouts_and_track_emissions():
          return perform_rollouts()

        all_rewards = perform_rollouts_and_track_emissions()
      else:
        all_rewards = perform_rollouts()

      metric_results['rollout_returns'] = dict(values=all_rewards,
                                               mean=np.mean(all_rewards),
                                               std=np.std(all_rewards))
      rollout_data_queue.put(metric_results)
    participant_event.clear()
    for profiler_event in profiler_events:
      profiler_event.clear()

    print('Finished inference. Now saving')
    with open(os.path.join(self.metric_values_dir,
                           'inference_metrics_results.json'),
              'w') as f:
      json.dump(metric_results, f)

  def _run_inference_reliability_metrics(self, values=None):
    metrics = []
    for metric in self.reliability_metrics:
      if metric == ReliabilityMetrics.MadAcrossRollouts:
        metrics.append(MadAcrossRollouts())
      elif metric == ReliabilityMetrics.IqrAcrossRollouts:
        metrics.append(IqrAcrossRollouts())
      elif metric == ReliabilityMetrics.UpperCVaRAcrossRollouts:
        metrics.append(UpperCVaRAcrossRollouts())
      elif metric == ReliabilityMetrics.LowerCVaRAcrossRollouts:
        metrics.append(LowerCVaRAcrossRollouts())
      elif metric == ReliabilityMetrics.StddevAcrossRollouts:
        metrics.append(StddevAcrossRollouts())
      elif metric == ReliabilityMetrics.UpperCVaRAcrossRollouts:
        metrics.append(UpperCVaRAcrossRollouts())
      elif metric == ReliabilityMetrics.LowerCVaRAcrossRollouts:
        metrics.append(LowerCVaRAcrossRollouts())
      else:
        raise ValueError(f'Invalid metric: {metric}')
    with open(os.path.join(self.metric_values_dir, 'rollouts.csv'), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['episode_num', 'reward'])
      for i, value in enumerate(values):
        writer.writerow([i, value])

    evaluator = Evaluator(metrics=metrics, )
    reliability_metrics = evaluator.evaluate(
        run_paths=[os.path.join(self.metric_values_dir, 'rollouts.csv')], )
    self.metrics_results.update(reliability_metrics)

  def _run_train_reliability_metrics(self):
    # TODO make sure to write gin config for metric parameters
    metrics = []
    for metric in self.reliability_metrics:
      if metric == ReliabilityMetrics.IqrWithinRuns:
        metrics.append(IqrWithinRuns())
      elif metric == ReliabilityMetrics.IqrAcrossRuns:
        metrics.append(IqrAcrossRuns())
      elif metric == ReliabilityMetrics.LowerCVaROnDiffs:
        metrics.append(LowerCVaROnDiffs())
      elif metric == ReliabilityMetrics.LowerCVaROnDrawdown:
        metrics.append(LowerCVaROnDiffs())
      elif metric == ReliabilityMetrics.LowerCVarOnAcross:
        metrics.append(LowerCVaROnAcross())
      elif metric == ReliabilityMetrics.MedianPerfDuringTraining:
        metrics.append(MedianPerfDuringTraining())
      else:
        raise ValueError(f'Invalid metric: {metric}')

    logging.info(f'Running reliability metrics: {metrics}')
    logging.info(f'Logging to {self.metric_values_dir}')

    if self.train_logs_dirs:
      run_paths = self.train_logs_dirs
      logging.info(f'Found {len(run_paths)} runs in {self.train_logs_dirs}')
      logging.info(f'Run paths: {run_paths}')

      evaluator = Evaluator(metrics=metrics, )
      reliability_metrics = evaluator.evaluate(run_paths=run_paths, )
    else:
      logging.warning(f'No runs found in {self.train_logs_dirs}')
      reliability_metrics = {}
    self.metrics_results.update(reliability_metrics)

  def _run_training_benchmark(self):
    if not self.run_offline_metrics_only:
      # Need a participant event to signal to profilers
      participant_started_event = multiprocessing.Event()
      profilers_started_events = [multiprocessing.Event() for _ in
                                  self.profilers]

      participant_process = self.mp_context.Process(target=self._train,
                                                    args=(
                                                        participant_started_event,
                                                        profilers_started_events))
      participant_process.start()
      profilers = _start_profilers(profilers=self.profilers,
                                   participant_event=participant_started_event,
                                   profiler_events=profilers_started_events,
                                   participant_process=participant_process,
                                   log_dir=self.root_dir,
                                   mp_context=self.mp_context)
      logging.info(f'Participant module process ID: {participant_process.pid}')
      participant_process.join()
      logging.info(
          f'Participant module process {participant_process.pid} finished')
      if participant_process.is_alive():
        logging.error('Participant process is still running')
      elif participant_process.exitcode != 0:
        logging.error(
            f'Participant process exited with code {participant_process.exitcode}')
      else:
        logging.info(f'Participant process {participant_process.pid} finished')

      for profiler in profilers:
        profiler.join()
        if profiler.is_alive():
          logging.error(f'Profiler process {profiler.pid} is still running')
        elif profiler.exitcode != 0:
          logging.error(
              f'Profiler process {profiler.pid} exited with code {profiler.exitcode}')
        else:
          logging.info(f'Profiler process {profiler.pid} finished')

    if self.reliability_metrics:
      self._run_train_reliability_metrics()

      ##################################################
      # Save raw metrics to disk, and plot results
      ##################################################

      with open(os.path.join(self.metric_values_dir, 'metric_results.json'),
                'w') as f:
        json.dump(self.metrics_results, f)

      # Plot metrics and save to file
      if self.plot_metrics:
        for profiler_object in self.profilers:
          title, fig = profiler_object.plot_results()
          plt.savefig(os.path.join(self.metric_values_dir, f'{title}.png'))
        self._plot_metrics()

  def _run_inference_benchmark(self):
    if not self.run_offline_metrics_only:
      env = self.create_domain()
      inference_data = self._get_observation_data(env)

      # Need a participant event to signal to profilers
      participant_started_event = multiprocessing.Event()
      profilers_started_events = [multiprocessing.Event() for _ in
                                  self.profilers]

      rollout_data_queue = self.mp_context.Queue()
      participant_process = self.mp_context.Process(target=self._inference,
                                                    args=(
                                                        participant_started_event,
                                                        profilers_started_events,
                                                        inference_data,
                                                        rollout_data_queue,
                                                    ))

      participant_process.start()
      profilers = _start_profilers(profilers=self.profilers,
                                   participant_event=participant_started_event,
                                   profiler_events=profilers_started_events,
                                   participant_process=participant_process,
                                   log_dir=self.root_dir,
                                   mp_context=self.mp_context)

      logging.info(f'Participant module process ID: {participant_process.pid}')
      participant_process.join()
      logging.info(
          f'Participant module process {participant_process.pid} finished')
      if participant_process.is_alive():
        logging.error('Participant process is still running')
      elif participant_process.exitcode != 0:
        logging.error(
            f'Participant process exited with code {participant_process.exitcode}')
      else:
        logging.info(f'Participant process {participant_process.pid} finished')

      for profiler in profilers:
        profiler.join()
        if profiler.is_alive():
          logging.error(f'Profiler process {profiler.pid} is still running')
        elif profiler.exitcode != 0:
          logging.error(
              f'Profiler process {profiler.pid} exited with code {profiler.exitcode}')
        else:
          logging.info(f'Profiler process {profiler.pid} finished')

  def run_benchmark(self):
    self.gin_config_str = gin.config_str()  # save gin configs for multiprocessing
    if self.mode == BenchmarkMode.TRAIN:
      self._run_training_benchmark()
    elif self.mode == BenchmarkMode.INFERENCE:
      self._run_inference_benchmark()
    else:
      raise ValueError('Benchmark mode must be either train or inference')
